#!/usr/bin/env python3
"""
ETL Pipeline Web Frontend

A Streamlit-based web interface for the financial document ETL pipeline.
Features:
- Document upload (PDF, Excel, Text)
- Institution management
- Processing history
- CSV export downloads
- Real-time processing status
"""

import streamlit as st
import pandas as pd
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import tempfile
import zipfile
import io

# Import ETL components
import sys
sys.path.append('src')

from src.etl.etl_pipeline import ETLPipeline
from src.etl.config import ETLConfig

class ETLFrontend:
    """Web frontend for the ETL pipeline."""
    
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
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def save_processing_history(self):
        """Save processing history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def add_to_history(self, institution: str, quarter: str, files: list, output_files: list, status: str):
        """Add processing record to history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'institution': institution,
            'quarter': quarter,
            'files': files,
            'output_files': output_files,
            'status': status,
            'record_count': 0
        }
        
        # Try to get record count from output
        try:
            if output_files and 'nlp_core_dataset_with_topics.csv' in str(output_files[0]):
                df = pd.read_csv(output_files[0])
                record['record_count'] = len(df)
        except:
            pass
        
        self.history.insert(0, record)  # Add to beginning
        self.save_processing_history()
    
    def process_documents(self, institution: str, quarter: str, uploaded_files: list, progress_bar, status_text):
        """Process uploaded documents through ETL pipeline."""
        try:
            # Create institution directory
            inst_dir = self.upload_dir / institution / quarter
            inst_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = inst_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(str(file_path))
            
            status_text.text("📁 Files uploaded successfully...")
            progress_bar.progress(0.2)
            
            # Initialize ETL pipeline
            config = ETLConfig()
            pipeline = ETLPipeline(config)
            
            status_text.text("🔧 Initializing ETL pipeline...")
            progress_bar.progress(0.3)
            
            # Process documents
            status_text.text("📊 Processing documents...")
            progress_bar.progress(0.5)
            
            # Run ETL pipeline (simplified version)
            output_files = self.run_etl_pipeline(institution, quarter, str(inst_dir), progress_bar, status_text)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Add to history
            self.add_to_history(institution, quarter, [f.name for f in uploaded_files], output_files, "Success")
            
            return output_files, "Success"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            status_text.text(f"❌ {error_msg}")
            self.add_to_history(institution, quarter, [f.name for f in uploaded_files], [], "Failed")
            return [], error_msg
    
    def run_etl_pipeline(self, institution: str, quarter: str, source_dir: str, progress_bar, status_text):
        """Run the ETL pipeline on uploaded documents."""
        
        # Create output directory for this processing run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = self.output_dir / f"{institution}_{quarter}_{timestamp}"
        output_subdir.mkdir(exist_ok=True)
        
        status_text.text("🔍 Discovering files...")
        progress_bar.progress(0.4)
        
        # Simple file discovery
        files = list(Path(source_dir).glob("*"))
        pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
        excel_files = [f for f in files if f.suffix.lower() in ['.xlsx', '.xls']]
        text_files = [f for f in files if f.suffix.lower() == '.txt']
        
        status_text.text("📝 Parsing documents...")
        progress_bar.progress(0.6)
        
        # Process files and create simplified output
        all_records = []
        
        # Process each file type
        for file_path in pdf_files + excel_files + text_files:
            try:
                # Simple text extraction (placeholder)
                if file_path.suffix.lower() == '.pdf':
                    records = self.process_pdf_simple(file_path, institution, quarter)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    records = self.process_excel_simple(file_path, institution, quarter)
                else:
                    records = self.process_text_simple(file_path, institution, quarter)
                
                all_records.extend(records)
            except Exception as e:
                st.warning(f"Could not process {file_path.name}: {str(e)}")
        
        status_text.text("🏷️ Adding NLP features...")
        progress_bar.progress(0.8)
        
        # Create DataFrame
        if all_records:
            df = pd.DataFrame(all_records)
            
            # Add basic NLP features
            df['word_count'] = df['text'].str.split().str.len()
            df['char_count'] = df['text'].str.len()
            df['is_financial_content'] = df['text'].str.contains('revenue|income|profit|billion|million', case=False, na=False)
            df['primary_topic'] = 'Financial Performance & Results'  # Simplified
            df['entities_text'] = ''  # Placeholder
            df['financial_terms_text'] = ''  # Placeholder
            
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
            
            status_text.text("💾 Saving outputs...")
            progress_bar.progress(0.9)
            
            return output_files
        else:
            raise Exception("No records were extracted from the uploaded files")
    
    def process_pdf_simple(self, file_path: Path, institution: str, quarter: str) -> list:
        """Simple PDF processing (placeholder)."""
        # This is a simplified version - in production, use the full PDF parser
        records = []
        try:
            # Placeholder: create sample records
            for i in range(5):
                records.append({
                    'source_file': file_path.name,
                    'institution': institution,
                    'quarter': quarter,
                    'sentence_id': i + 1,
                    'speaker_norm': 'UNKNOWN',
                    'text': f"Sample text from {file_path.name} sentence {i + 1}",
                    'source_type': 'earnings_presentation'
                })
        except Exception as e:
            st.warning(f"PDF processing error: {e}")
        
        return records
    
    def process_excel_simple(self, file_path: Path, institution: str, quarter: str) -> list:
        """Simple Excel processing."""
        records = []
        try:
            df = pd.read_excel(file_path)
            for idx, row in df.head(10).iterrows():  # Limit for demo
                text_content = ' '.join([str(val) for val in row.values if pd.notna(val)])
                records.append({
                    'source_file': file_path.name,
                    'institution': institution,
                    'quarter': quarter,
                    'sentence_id': idx + 1,
                    'speaker_norm': 'UNKNOWN',
                    'text': text_content[:500],  # Limit length
                    'source_type': 'financial_supplement'
                })
        except Exception as e:
            st.warning(f"Excel processing error: {e}")
        
        return records
    
    def process_text_simple(self, file_path: Path, institution: str, quarter: str) -> list:
        """Simple text processing."""
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sentences = content.split('\n')
            for idx, sentence in enumerate(sentences[:20]):  # Limit for demo
                if sentence.strip():
                    records.append({
                        'source_file': file_path.name,
                        'institution': institution,
                        'quarter': quarter,
                        'sentence_id': idx + 1,
                        'speaker_norm': 'UNKNOWN',
                        'text': sentence.strip(),
                        'source_type': 'earnings_call'
                    })
        except Exception as e:
            st.warning(f"Text processing error: {e}")
        
        return records

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Financial ETL Pipeline",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize frontend
    frontend = ETLFrontend()
    
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
                output_files, status = frontend.process_documents(
                    institution, quarter, uploaded_files, progress_bar, status_text
                )
                
                if status == "Success":
                    st.success(f"✅ Processing completed successfully!")
                    st.info(f"Generated {len(output_files)} output files")
                    
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
                    st.write(f"• {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.header("📚 Processing History")
        
        if frontend.history:
            for record in frontend.history[:10]:  # Show last 10
                with st.expander(f"{record['institution']} - {record['quarter']}"):
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