#!/usr/bin/env python3
"""
Simplified ETL Pipeline Web Frontend

A Streamlit-based web interface that doesn't depend on complex ETL configurations.
Features:
- Document upload (PDF, Excel, Text)
- Institution management
- Processing history
- CSV export downloads
- Simplified processing pipeline
"""

import streamlit as st
import pandas as pd
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import tempfile
import re
import PyPDF2
import openpyxl

class SimpleETLFrontend:
    """Simplified web frontend for the ETL pipeline."""
    
    def __init__(self):
        """Initialize the frontend."""
        self.setup_directories()
        self.load_processing_history()
    
    def setup_directories(self):
        """Setup required directories."""
        self.upload_dir = Path("uploads")
        self.output_dir = Path("outputs")
        self.history_file = Path("processing_history.json")
        
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_processing_history(self):
        """Load processing history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = []
        else:
            self.history = []
    
    def save_processing_history(self):
        """Save processing history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def add_to_history(self, institution: str, quarter: str, files: list, output_files: list, status: str, record_count: int = 0):
        """Add processing record to history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'institution': institution,
            'quarter': quarter,
            'files': files,
            'output_files': output_files,
            'status': status,
            'record_count': record_count
        }
        
        self.history.insert(0, record)  # Add to beginning
        if len(self.history) > 50:  # Keep only last 50 records
            self.history = self.history[:50]
        self.save_processing_history()
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            st.warning(f"Could not extract text from {file_path.name}: {str(e)}")
            return ""
    
    def extract_text_from_excel(self, file_path: Path) -> str:
        """Extract text from Excel file."""
        try:
            df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"Sheet: {sheet_name}\n"
                for _, row in sheet_df.iterrows():
                    row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
                    if row_text.strip():
                        text += row_text + "\n"
                text += "\n"
            return text
        except Exception as e:
            st.warning(f"Could not extract text from {file_path.name}: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e2:
                st.warning(f"Could not read text file {file_path.name}: {str(e2)}")
                return ""
    
    def segment_text(self, text: str) -> list:
        """Segment text into sentences."""
        # Simple sentence segmentation
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def identify_speaker(self, text: str) -> str:
        """Simple speaker identification."""
        # Look for speaker patterns
        speaker_patterns = [
            r'^([A-Z][A-Z\s]+):\s*',  # ALL CAPS NAME:
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+):\s*',  # First Last:
            r'(CEO|CFO|Chief Executive|Chief Financial)',  # Titles
        ]
        
        for pattern in speaker_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return 'UNKNOWN'
    
    def classify_document_type(self, filename: str, text: str) -> str:
        """Classify document type based on filename and content."""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if 'presentation' in filename_lower or 'slides' in filename_lower:
            return 'earnings_presentation'
        elif 'supplement' in filename_lower or 'financial' in filename_lower:
            return 'financial_supplement'
        elif 'transcript' in filename_lower or 'call' in filename_lower:
            return 'earnings_call'
        elif 'earnings' in text_lower and 'call' in text_lower:
            return 'earnings_call'
        elif 'financial' in text_lower and ('results' in text_lower or 'performance' in text_lower):
            return 'financial_results'
        else:
            return 'other'
    
    def add_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic NLP features to the dataframe."""
        # Text metrics
        df['word_count'] = df['text'].str.split().str.len()
        df['char_count'] = df['text'].str.len()
        
        # Content flags
        financial_terms = ['revenue', 'income', 'profit', 'loss', 'earnings', 'billion', 'million', 
                          'eps', 'roe', 'capital', 'assets', 'liabilities', 'equity']
        financial_pattern = '|'.join(financial_terms)
        df['is_financial_content'] = df['text'].str.contains(financial_pattern, case=False, na=False)
        
        # Speaker analysis
        df['is_management'] = df['speaker_norm'].str.contains('CEO|CFO|Chief|Executive|Financial', case=False, na=False)
        df['is_analyst'] = df['speaker_norm'].str.contains('Analyst', case=False, na=False)
        df['is_named_speaker'] = df['speaker_norm'] != 'UNKNOWN'
        
        # Simple topic assignment
        def assign_topic(text):
            text_lower = str(text).lower()
            if any(term in text_lower for term in ['revenue', 'income', 'growth']):
                return 'Revenue & Income Growth'
            elif any(term in text_lower for term in ['risk', 'credit', 'provision']):
                return 'Risk Management & Credit'
            elif any(term in text_lower for term in ['capital', 'regulatory', 'ratio']):
                return 'Capital & Regulatory Metrics'
            elif any(term in text_lower for term in ['strategy', 'outlook', 'future']):
                return 'Business Strategy & Outlook'
            elif any(term in text_lower for term in ['cost', 'efficiency', 'expense']):
                return 'Operational Efficiency & Costs'
            else:
                return 'Banking Operations & Services'
        
        df['primary_topic'] = df['text'].apply(assign_topic)
        
        # Topic flags
        df['has_financial_topic'] = df['primary_topic'].str.contains('Revenue|Capital|Risk', case=False, na=False)
        df['has_strategy_topic'] = df['primary_topic'].str.contains('Strategy|Outlook', case=False, na=False)
        df['has_performance_topic'] = df['primary_topic'].str.contains('Revenue|Growth|Performance', case=False, na=False)
        
        return df
    
    def process_documents(self, institution: str, quarter: str, uploaded_files: list, progress_bar, status_text):
        """Process uploaded documents through simplified ETL pipeline."""
        try:
            # Create institution directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            inst_dir = self.upload_dir / institution / quarter / timestamp
            inst_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded files and extract text
            all_records = []
            file_names = []
            
            status_text.text("📁 Saving uploaded files...")
            progress_bar.progress(0.1)
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = inst_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_names.append(uploaded_file.name)
                
                status_text.text(f"📄 Processing {uploaded_file.name}...")
                progress_bar.progress(0.2 + (i * 0.3 / len(uploaded_files)))
                
                # Extract text based on file type
                if uploaded_file.name.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                    text = self.extract_text_from_excel(file_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    text = self.extract_text_from_txt(file_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                if not text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name}")
                    continue
                
                # Segment text into sentences
                sentences = self.segment_text(text)
                
                # Create records
                doc_type = self.classify_document_type(uploaded_file.name, text)
                
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
                raise Exception("No text content could be extracted from the uploaded files")
            
            status_text.text("🔧 Creating structured dataset...")
            progress_bar.progress(0.6)
            
            # Create DataFrame
            df = pd.DataFrame(all_records)
            
            # Add NLP features
            status_text.text("🧠 Adding NLP features...")
            progress_bar.progress(0.8)
            
            df = self.add_nlp_features(df)
            
            # Create output directory
            output_subdir = self.output_dir / f"{institution}_{quarter}_{timestamp}"
            output_subdir.mkdir(exist_ok=True)
            
            # Save outputs
            output_files = []
            
            # Core dataset
            core_file = output_subdir / f"{institution}_{quarter}_nlp_core_dataset.csv"
            df.to_csv(core_file, index=False)
            output_files.append(str(core_file))
            
            # Financial subset
            financial_df = df[df['is_financial_content']]
            if len(financial_df) > 0:
                financial_file = output_subdir / f"{institution}_{quarter}_financial_dataset.csv"
                financial_df.to_csv(financial_file, index=False)
                output_files.append(str(financial_file))
            
            # Speaker subset (named speakers only)
            speaker_df = df[df['is_named_speaker']]
            if len(speaker_df) > 0:
                speaker_file = output_subdir / f"{institution}_{quarter}_speaker_dataset.csv"
                speaker_df.to_csv(speaker_file, index=False)
                output_files.append(str(speaker_file))
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(1.0)
            
            # Add to history
            self.add_to_history(institution, quarter, file_names, output_files, "Success", len(df))
            
            return output_files, "Success", len(df)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            status_text.text(f"❌ {error_msg}")
            self.add_to_history(institution, quarter, file_names if 'file_names' in locals() else [], [], "Failed", 0)
            return [], error_msg, 0

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Financial ETL Pipeline",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize frontend
    frontend = SimpleETLFrontend()
    
    # Header
    st.title("📊 Financial Document ETL Pipeline")
    st.markdown("Transform unstructured financial documents into structured NLP-ready datasets")
    
    # Sidebar
    st.sidebar.header("🏛️ Institution Processing")
    
    # Institution input
    institution = st.sidebar.text_input(
        "Institution Name",
        placeholder="e.g., JPMorgan Chase, Bank of America",
        help="Enter the financial institution name"
    )
    
    # Quarter input
    quarter_options = [
        "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025",
        "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"
    ]
    quarter = st.sidebar.selectbox("Quarter", quarter_options)
    
    # File upload
    st.sidebar.subheader("📁 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=['pdf', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF presentations, Excel supplements, or text transcripts"
    )
    
    # Process button
    process_button = st.sidebar.button(
        "🚀 Process Documents",
        disabled=not (institution and uploaded_files),
        help="Start ETL processing"
    )
    
    # Main content area
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
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process documents
                output_files, status, record_count = frontend.process_documents(
                    institution, quarter, uploaded_files, progress_bar, status_text
                )
                
                if status == "Success":
                    st.success(f"✅ Processing completed successfully!")
                    st.info(f"Generated {record_count:,} records in {len(output_files)} files")
                    
                    # Download buttons
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
            
            # Show upload preview
            if uploaded_files:
                st.subheader("📁 Uploaded Files")
                for file in uploaded_files:
                    file_size = f"{file.size:,} bytes" if file.size < 1024*1024 else f"{file.size/(1024*1024):.1f} MB"
                    st.write(f"• {file.name} ({file_size})")
    
    with col2:
        st.header("📚 Processing History")
        
        if frontend.history:
            for record in frontend.history[:10]:  # Show last 10
                status_icon = "✅" if record['status'] == 'Success' else "❌"
                with st.expander(f"{status_icon} {record['institution']} - {record['quarter']}"):
                    st.write(f"**Date:** {record['timestamp'][:19]}")
                    st.write(f"**Status:** {record['status']}")
                    st.write(f"**Files:** {len(record['files'])}")
                    if record['record_count']:
                        st.write(f"**Records:** {record['record_count']:,}")
                    
                    if record['files']:
                        st.write("**Uploaded Files:**")
                        for file in record['files']:
                            st.write(f"• {file}")
        else:
            st.info("No processing history yet")
    
    # Footer
    st.markdown("---")
    st.markdown("**ETL Pipeline Features:** Document parsing • NLP processing • Topic modeling • Entity recognition • Financial term tagging")

if __name__ == "__main__":
    main()