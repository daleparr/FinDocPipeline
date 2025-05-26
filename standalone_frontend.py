#!/usr/bin/env python3
"""
Standalone ETL Pipeline Web Frontend

A completely self-contained Streamlit web interface that doesn't depend on 
any existing ETL modules or configurations.
"""

import streamlit as st
import pandas as pd
import os
import json
import re
from datetime import datetime
from pathlib import Path
import tempfile

# Only use standard libraries and basic packages
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

class StandaloneETL:
    """Completely standalone ETL processor."""
    
    def __init__(self):
        """Initialize the standalone ETL."""
        self.setup_directories()
        self.load_history()
    
    def setup_directories(self):
        """Setup required directories."""
        self.upload_dir = Path("frontend_uploads")
        self.output_dir = Path("frontend_outputs")
        self.history_file = Path("frontend_history.json")
        
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_history(self):
        """Load processing history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = []
        else:
            self.history = []
    
    def save_history(self):
        """Save processing history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def extract_pdf_text(self, file_path):
        """Extract text from PDF."""
        if not PDF_AVAILABLE:
            return "PDF processing not available. Install PyPDF2."
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_excel_text(self, file_path):
        """Extract text from Excel."""
        if not EXCEL_AVAILABLE:
            return "Excel processing not available. Install openpyxl."
        
        try:
            text = ""
            df_dict = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in df_dict.items():
                text += f"Sheet: {sheet_name}\n"
                for _, row in df.iterrows():
                    row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
                    if row_text.strip():
                        text += row_text + "\n"
                text += "\n"
            return text
        except Exception as e:
            return f"Error reading Excel: {str(e)}"
    
    def extract_text_file(self, file_path):
        """Extract text from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading text file: {str(e)}"
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    
    def segment_sentences(self, text):
        """Simple sentence segmentation."""
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = re.sub(r'\s+', ' ', sentence)  # Normalize whitespace
            if len(sentence) > 10:  # Minimum length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def identify_speaker(self, text):
        """Simple speaker identification."""
        # Look for speaker patterns at start of text
        patterns = [
            r'^([A-Z][A-Z\s]+):\s*',  # ALL CAPS:
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+):\s*',  # First Last:
            r'^(CEO|CFO|Chief Executive|Chief Financial)',  # Titles
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return 'UNKNOWN'
    
    def classify_document(self, filename, text):
        """Classify document type."""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if 'presentation' in filename_lower:
            return 'earnings_presentation'
        elif 'supplement' in filename_lower or 'financial' in filename_lower:
            return 'financial_supplement'
        elif 'transcript' in filename_lower or 'call' in filename_lower:
            return 'earnings_call'
        elif 'earnings' in text_lower and 'call' in text_lower:
            return 'earnings_call'
        else:
            return 'other'
    
    def add_nlp_features(self, df):
        """Add comprehensive NLP features including financial terms and figures."""
        # Basic text metrics
        df['word_count'] = df['text'].str.split().str.len()
        df['char_count'] = df['text'].str.len()
        
        # Comprehensive financial terms list
        financial_terms_list = [
            'revenue', 'income', 'profit', 'earnings', 'billion', 'million',
            'eps', 'capital', 'assets', 'growth', 'performance', 'margin',
            'return', 'yield', 'dividend', 'interest', 'loan', 'credit',
            'deposit', 'fee', 'commission', 'expense', 'cost', 'investment',
            'portfolio', 'risk', 'regulatory', 'compliance', 'basel',
            'tier', 'ratio', 'liquidity', 'solvency', 'provision'
        ]
        
        # Extract financial terms from each text
        def extract_financial_terms(text):
            text_lower = str(text).lower()
            found_terms = []
            for term in financial_terms_list:
                if term in text_lower:
                    found_terms.append(term)
            return '|'.join(found_terms) if found_terms else ''
        
        df['all_financial_terms'] = df['text'].apply(extract_financial_terms)
        
        # Extract financial figures (numbers with financial context)
        def extract_financial_figures(text):
            import re
            text_str = str(text)
            
            # Patterns for financial figures
            patterns = [
                r'\$[\d,]+\.?\d*\s*(?:billion|million|thousand|B|M|K)?',  # Dollar amounts
                r'[\d,]+\.?\d*\s*(?:billion|million|thousand|percent|%|basis points|bps)',  # Numbers with units
                r'[\d,]+\.?\d*\s*(?:dollars|cents)',  # Dollar/cent amounts
                r'(?:approximately|about|around|roughly)\s*[\d,]+\.?\d*',  # Approximate figures
            ]
            
            figures = []
            for pattern in patterns:
                matches = re.findall(pattern, text_str, re.IGNORECASE)
                figures.extend(matches)
            
            return '|'.join(figures) if figures else ''
        
        df['financial_figures'] = df['text'].apply(extract_financial_figures)
        
        # Create financial_figures_text (same as financial_figures for compatibility)
        df['financial_figures_text'] = df['financial_figures']
        
        # Classify actual vs projected financial data
        def classify_actual_vs_projection(text):
            text_lower = str(text).lower()
            
            # Projection indicators
            projection_terms = [
                'expect', 'forecast', 'project', 'anticipate', 'estimate',
                'guidance', 'outlook', 'target', 'goal', 'plan', 'intend',
                'will be', 'should be', 'likely to', 'going forward',
                'next quarter', 'next year', 'future', 'upcoming'
            ]
            
            # Actual/historical indicators
            actual_terms = [
                'reported', 'achieved', 'delivered', 'recorded', 'posted',
                'was', 'were', 'had', 'generated', 'earned', 'realized',
                'last quarter', 'previous', 'year-over-year', 'compared to'
            ]
            
            projection_score = sum(1 for term in projection_terms if term in text_lower)
            actual_score = sum(1 for term in actual_terms if term in text_lower)
            
            if projection_score > actual_score:
                return 'projection'
            elif actual_score > projection_score:
                return 'actual'
            else:
                return 'unclear'
        
        df['data_type'] = df['text'].apply(classify_actual_vs_projection)
        
        # Boolean flags for data type
        df['is_actual_data'] = df['data_type'] == 'actual'
        df['is_projection_data'] = df['data_type'] == 'projection'
        
        # Financial content detection (enhanced)
        df['is_financial_content'] = (df['all_financial_terms'] != '') | (df['financial_figures'] != '')
        
        # Speaker analysis
        df['is_management'] = df['speaker_norm'].str.contains('CEO|CFO|Chief', case=False, na=False)
        df['is_analyst'] = df['speaker_norm'].str.contains('Analyst', case=False, na=False)
        df['is_named_speaker'] = df['speaker_norm'] != 'UNKNOWN'
        
        # Enhanced topic assignment
        def assign_topic(text):
            text_lower = str(text).lower()
            if any(term in text_lower for term in ['revenue', 'income', 'growth', 'earnings']):
                return 'Revenue & Growth'
            elif any(term in text_lower for term in ['risk', 'credit', 'provision', 'loss']):
                return 'Risk Management'
            elif any(term in text_lower for term in ['capital', 'regulatory', 'basel', 'tier']):
                return 'Capital & Regulatory'
            elif any(term in text_lower for term in ['strategy', 'outlook', 'guidance', 'plan']):
                return 'Strategy & Outlook'
            elif any(term in text_lower for term in ['cost', 'efficiency', 'expense', 'margin']):
                return 'Operational Efficiency'
            elif any(term in text_lower for term in ['digital', 'technology', 'innovation']):
                return 'Digital & Technology'
            else:
                return 'General Banking'
        
        df['primary_topic'] = df['text'].apply(assign_topic)
        
        # Enhanced topic flags
        df['has_financial_topic'] = df['primary_topic'].str.contains('Revenue|Capital|Risk', na=False)
        df['has_strategy_topic'] = df['primary_topic'].str.contains('Strategy', na=False)
        df['has_operational_topic'] = df['primary_topic'].str.contains('Operational', na=False)
        
        return df
    
    def process_files(self, institution, quarter, uploaded_files, progress_callback=None):
        """Process uploaded files."""
        try:
            # Create processing directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            process_dir = self.upload_dir / f"{institution}_{quarter}_{timestamp}"
            process_dir.mkdir(exist_ok=True)
            
            all_records = []
            file_names = []
            
            if progress_callback:
                progress_callback(0.1, "Saving uploaded files...")
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = process_dir / uploaded_file.name
                
                # Save file
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                file_names.append(uploaded_file.name)
                
                if progress_callback:
                    progress_callback(0.2 + (i * 0.4 / len(uploaded_files)), 
                                    f"Processing {uploaded_file.name}...")
                
                # Extract text based on file type
                if uploaded_file.name.lower().endswith('.pdf'):
                    text = self.extract_pdf_text(file_path)
                elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                    text = self.extract_excel_text(file_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    text = self.extract_text_file(file_path)
                else:
                    text = f"Unsupported file type: {uploaded_file.name}"
                
                if not text or text.startswith("Error") or len(text.strip()) < 10:
                    st.warning(f"Could not extract meaningful text from {uploaded_file.name}")
                    continue
                
                # Segment into sentences
                sentences = self.segment_sentences(text)
                
                # Create records
                doc_type = self.classify_document(uploaded_file.name, text)
                
                for idx, sentence in enumerate(sentences):
                    speaker = self.identify_speaker(sentence)
                    
                    record = {
                        'source_file': uploaded_file.name,
                        'institution': institution,
                        'quarter': quarter,
                        'sentence_id': idx + 1,
                        'speaker_norm': speaker,
                        'text': sentence,
                        'source_type': doc_type,
                        'call_id': f"{institution}_{quarter}_{timestamp}",
                        'file_path': str(file_path),
                        'processing_date': datetime.now().isoformat()
                    }
                    all_records.append(record)
            
            if not all_records:
                raise Exception("No text content could be extracted from uploaded files")
            
            if progress_callback:
                progress_callback(0.7, "Creating structured dataset...")
            
            # Create DataFrame
            df = pd.DataFrame(all_records)
            
            # Add NLP features
            if progress_callback:
                progress_callback(0.8, "Adding NLP features...")
            
            df = self.add_nlp_features(df)
            
            # Save outputs
            output_dir = self.output_dir / f"{institution}_{quarter}_{timestamp}"
            output_dir.mkdir(exist_ok=True)
            
            output_files = []
            
            # Core dataset
            core_file = output_dir / f"{institution}_{quarter}_core_dataset.csv"
            df.to_csv(core_file, index=False)
            output_files.append(str(core_file))
            
            # Financial subset
            financial_df = df[df['is_financial_content']]
            if len(financial_df) > 0:
                financial_file = output_dir / f"{institution}_{quarter}_financial_dataset.csv"
                financial_df.to_csv(financial_file, index=False)
                output_files.append(str(financial_file))
            
            if progress_callback:
                progress_callback(1.0, "Processing complete!")
            
            # Add to history
            self.history.insert(0, {
                'timestamp': datetime.now().isoformat(),
                'institution': institution,
                'quarter': quarter,
                'files': file_names,
                'output_files': output_files,
                'status': 'Success',
                'record_count': len(df)
            })
            self.save_history()
            
            return output_files, len(df), "Success"
            
        except Exception as e:
            error_msg = str(e)
            
            # Add failed record to history
            self.history.insert(0, {
                'timestamp': datetime.now().isoformat(),
                'institution': institution,
                'quarter': quarter,
                'files': file_names if 'file_names' in locals() else [],
                'output_files': [],
                'status': 'Failed',
                'record_count': 0,
                'error': error_msg
            })
            self.save_history()
            
            return [], 0, error_msg

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Financial ETL Pipeline",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize ETL
    etl = StandaloneETL()
    
    # Header
    st.title("📊 Financial Document ETL Pipeline")
    st.markdown("**Standalone Version** - Transform financial documents into structured datasets")
    
    # Check dependencies
    missing_deps = []
    if not PDF_AVAILABLE:
        missing_deps.append("PyPDF2 (for PDF processing)")
    if not EXCEL_AVAILABLE:
        missing_deps.append("openpyxl (for Excel processing)")
    
    if missing_deps:
        st.warning(f"Optional dependencies missing: {', '.join(missing_deps)}")
        st.info("Install with: `pip install PyPDF2 openpyxl`")
    
    # Sidebar
    with st.sidebar:
        st.header("🏛️ Institution Processing")
        
        # Institution input
        institution = st.text_input(
            "Institution Name",
            placeholder="e.g., JPMorgan Chase",
            help="Enter the financial institution name"
        )
        
        # Quarter selection
        quarters = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025", 
                   "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
        quarter = st.selectbox("Quarter", quarters)
        
        # File upload
        st.subheader("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Upload financial documents"
        )
        
        # Process button
        process_button = st.button(
            "🚀 Process Documents",
            disabled=not (institution and uploaded_files),
            type="primary"
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Processing Status")
        
        if process_button:
            if not institution:
                st.error("Please enter an institution name")
            elif not uploaded_files:
                st.error("Please upload at least one document")
            else:
                # Processing
                st.info(f"Processing {len(uploaded_files)} documents for {institution} {quarter}...")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value, message):
                    progress_bar.progress(value)
                    status_text.text(message)
                
                # Process files
                output_files, record_count, status = etl.process_files(
                    institution, quarter, uploaded_files, update_progress
                )
                
                if status == "Success":
                    st.success(f"✅ Processing completed successfully!")
                    st.info(f"Generated {record_count:,} records in {len(output_files)} files")
                    
                    # Download section
                    st.subheader("📥 Download Results")
                    for output_file in output_files:
                        file_path = Path(output_file)
                        if file_path.exists():
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    label=f"📄 Download {file_path.name}",
                                    data=f.read(),
                                    file_name=file_path.name,
                                    mime="text/csv"
                                )
                else:
                    st.error(f"❌ Processing failed: {status}")
        
        else:
            st.info("👆 Upload documents and click 'Process Documents' to begin")
            
            if uploaded_files:
                st.subheader("📁 Uploaded Files")
                for file in uploaded_files:
                    size_mb = file.size / (1024 * 1024)
                    st.write(f"• {file.name} ({size_mb:.1f} MB)")
    
    with col2:
        st.header("📚 Processing History")
        
        if etl.history:
            for record in etl.history[:8]:  # Show last 8
                status_icon = "✅" if record['status'] == 'Success' else "❌"
                
                with st.expander(f"{status_icon} {record['institution']} - {record['quarter']}"):
                    st.write(f"**Date:** {record['timestamp'][:19]}")
                    st.write(f"**Status:** {record['status']}")
                    st.write(f"**Files:** {len(record['files'])}")
                    
                    if record['status'] == 'Success':
                        st.write(f"**Records:** {record['record_count']:,}")
                    else:
                        if 'error' in record:
                            st.write(f"**Error:** {record['error'][:100]}...")
                    
                    if record['files']:
                        st.write("**Files:**")
                        for file in record['files']:
                            st.write(f"• {file}")
        else:
            st.info("No processing history yet")
    
    # Footer
    st.markdown("---")
    st.markdown("**Features:** PDF/Excel/Text parsing • NLP processing • Multi-institution support • CSV export")

if __name__ == "__main__":
    main()