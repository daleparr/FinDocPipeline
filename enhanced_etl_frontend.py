#!/usr/bin/env python3
"""
Enhanced BOE-ETL Frontend with Visual Data Extraction
===================================================

Streamlit frontend that demonstrates the enhanced visual data extraction
capabilities for financial presentations and earnings documents.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Add boe-etl to path
sys.path.insert(0, str(Path(__file__).parent / "boe-etl"))

# Import enhanced components
try:
    from boe_etl.parsers.enhanced_pdf_parser import EnhancedPDFParser
    from boe_etl.enhanced_schema_transformer import EnhancedSchemaTransformer
    ENHANCED_AVAILABLE = True
except ImportError as e:
    st.error(f"Enhanced components not available: {e}")
    ENHANCED_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Enhanced BOE-ETL with Visual Extraction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application."""
    
    # Header
    st.title("📊 Enhanced BOE-ETL Pipeline")
    st.subheader("Financial Document Processing with Visual Data Extraction")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Institution selection
        institution = st.selectbox(
            "Select Institution",
            ["JPMorgan Chase", "Bank of America", "Citigroup", "Wells Fargo", "Goldman Sachs", "Other"],
            index=0
        )
        
        if institution == "Other":
            institution = st.text_input("Enter Institution Name", "CustomBank")
        
        # Quarter selection
        quarter = st.selectbox(
            "Select Quarter",
            ["Q1_2025", "Q4_2024", "Q3_2024", "Q2_2024", "Q1_2024"],
            index=1
        )
        
        st.divider()
        
        # Enhanced processing options
        st.header("🔧 Processing Options")
        
        # Standard options
        extract_text = st.checkbox("Extract Text Content", value=True)
        extract_tables = st.checkbox("Extract Tables", value=True)
        
        # Enhanced visual options
        st.subheader("📊 Visual Extraction (NEW)")
        enable_visual = st.checkbox(
            "Enable Visual Data Extraction", 
            value=True,
            help="Extract data from charts, graphs, and visual elements"
        )
        
        if enable_visual:
            extract_charts = st.checkbox("Extract Chart Data", value=True)
            analyze_trends = st.checkbox("Analyze Trends & Patterns", value=True)
            detect_emphasis = st.checkbox("Detect Visual Emphasis", value=True)
            
            # Advanced options
            with st.expander("Advanced Visual Settings"):
                confidence_threshold = st.slider(
                    "Classification Confidence Threshold", 
                    0.5, 1.0, 0.7, 0.05
                )
                max_pages = st.number_input(
                    "Max Pages to Process (0 = all)", 
                    0, 100, 0
                )
        else:
            extract_charts = False
            analyze_trends = False
            detect_emphasis = False
            confidence_threshold = 0.7
            max_pages = 0
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Financial Documents",
            type=['pdf', 'xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Upload earnings presentations, financial reports, or transcripts"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            # Show file details
            for file in uploaded_files:
                st.write(f"📄 **{file.name}** ({file.size:,} bytes)")
    
    with col2:
        st.header("ℹ️ System Status")
        
        # System capabilities
        if ENHANCED_AVAILABLE:
            st.success("✅ Enhanced Visual Extraction Available")
            st.info("🔍 Chart Detection: Enabled")
            st.info("📈 Trend Analysis: Enabled") 
            st.info("🎯 Risk Assessment: Enabled")
        else:
            st.warning("⚠️ Enhanced Features Unavailable")
            st.info("📝 Text Extraction: Available")
        
        # Processing stats
        if 'processing_stats' in st.session_state:
            stats = st.session_state.processing_stats
            st.metric("Documents Processed", stats.get('documents', 0))
            st.metric("Charts Detected", stats.get('charts', 0))
            st.metric("Data Points Extracted", stats.get('data_points', 0))
    
    # Processing section
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        process_documents(
            uploaded_files, institution, quarter,
            enable_visual, extract_charts, analyze_trends,
            confidence_threshold, max_pages
        )
    
    # Results section
    if 'results' in st.session_state:
        display_results()

def process_documents(uploaded_files, institution, quarter, enable_visual, 
                     extract_charts, analyze_trends, confidence_threshold, max_pages):
    """Process uploaded documents with enhanced extraction."""
    
    if not ENHANCED_AVAILABLE:
        st.error("❌ Enhanced processing not available. Please check installation.")
        return
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("🔧 Initializing enhanced processors...")
        
        parser_config = {
            'enable_visual_extraction': enable_visual,
            'enable_chart_analysis': extract_charts,
            'enable_trend_detection': analyze_trends,
            'chart_extraction': {
                'classification': {'confidence_threshold': confidence_threshold}
            },
            'processing_options': {
                'max_pages': max_pages if max_pages > 0 else None
            }
        }
        
        parser = EnhancedPDFParser(institution, quarter, parser_config)
        transformer = EnhancedSchemaTransformer({
            'visual_processing': {
                'include_visual_text': True,
                'generate_descriptions': True,
                'min_confidence_threshold': confidence_threshold
            }
        })
        
        progress_bar.progress(0.1)
        
        # Process each file
        all_records = []
        processing_stats = {'documents': 0, 'charts': 0, 'data_points': 0}
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"📄 Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = Path(tmp_file.name)
            
            try:
                # Parse document
                if tmp_file_path.suffix.lower() == '.pdf':
                    parsed_data = parser.parse(tmp_file_path)
                    
                    # Update stats
                    processing_stats['documents'] += 1
                    visual_elements = parsed_data.get('visual_elements', {})
                    charts = visual_elements.get('charts', [])
                    processing_stats['charts'] += len(charts)
                    
                    for chart in charts:
                        processing_stats['data_points'] += len(chart.get('data_points', []))
                    
                    # Transform to records
                    records = transformer.transform_parsed_data(parsed_data, institution, quarter)
                    all_records.extend(records)
                    
                else:
                    st.warning(f"⚠️ File type {tmp_file_path.suffix} not yet supported for visual extraction")
                
            finally:
                # Clean up temp file
                if tmp_file_path.exists():
                    os.unlink(tmp_file_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Processing completed!")
        
        # Store results
        st.session_state.results = all_records
        st.session_state.processing_stats = processing_stats
        
        # Success message
        st.success(f"🎉 Successfully processed {len(uploaded_files)} document(s)!")
        st.info(f"📊 Generated {len(all_records)} total records ({processing_stats['charts']} charts detected)")
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)

def display_results():
    """Display processing results with enhanced visualizations."""
    
    st.header("📊 Processing Results")
    
    records = st.session_state.results
    stats = st.session_state.processing_stats
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(records))
    
    with col2:
        text_records = len([r for r in records if not r.get('visual_element_type')])
        st.metric("Text Records", text_records)
    
    with col3:
        visual_records = len([r for r in records if r.get('visual_element_type')])
        st.metric("Visual Records", visual_records)
    
    with col4:
        st.metric("Charts Detected", stats.get('charts', 0))
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Data", "📊 Visual Elements", "📈 Trends", "💾 Export"])
    
    with tab1:
        st.subheader("Complete Dataset")
        
        if records:
            df = pd.DataFrame(records)
            
            # Show key columns
            display_columns = [
                'bank', 'quarter', 'text', 'visual_element_type', 'chart_type',
                'trend_direction', 'financial_metric_type', 'extraction_confidence'
            ]
            
            available_columns = [col for col in display_columns if col in df.columns]
            
            st.dataframe(
                df[available_columns],
                use_container_width=True,
                height=400
            )
            
            # Show schema info
            with st.expander("📋 Schema Information"):
                st.write(f"**Total Columns**: {len(df.columns)}")
                st.write("**Column Names**:")
                for col in sorted(df.columns):
                    st.write(f"- {col}")
        else:
            st.info("No data to display")
    
    with tab2:
        st.subheader("Visual Elements Analysis")
        
        visual_records = [r for r in records if r.get('visual_element_type')]
        
        if visual_records:
            visual_df = pd.DataFrame(visual_records)
            
            # Chart type distribution
            if 'chart_type' in visual_df.columns:
                chart_counts = visual_df['chart_type'].value_counts()
                st.bar_chart(chart_counts)
            
            # Visual elements table
            visual_columns = [
                'visual_element_type', 'chart_type', 'trend_direction', 
                'trend_magnitude', 'extraction_confidence', 'text'
            ]
            available_visual_columns = [col for col in visual_columns if col in visual_df.columns]
            
            st.dataframe(
                visual_df[available_visual_columns],
                use_container_width=True
            )
        else:
            st.info("No visual elements detected")
    
    with tab3:
        st.subheader("Trend Analysis")
        
        trend_records = [r for r in records if r.get('trend_direction') and r.get('trend_direction') != 'stable']
        
        if trend_records:
            trend_df = pd.DataFrame(trend_records)
            
            # Trend distribution
            if 'trend_direction' in trend_df.columns:
                trend_counts = trend_df['trend_direction'].value_counts()
                st.bar_chart(trend_counts)
            
            # Risk indicators
            if 'risk_indicator_type' in trend_df.columns:
                risk_counts = trend_df['risk_indicator_type'].value_counts()
                st.write("**Risk Indicators**")
                st.bar_chart(risk_counts)
            
            # Detailed trends
            st.write("**Detailed Trend Analysis**")
            trend_columns = [
                'chart_type', 'trend_direction', 'trend_magnitude', 
                'risk_indicator_type', 'financial_metric_type'
            ]
            available_trend_columns = [col for col in trend_columns if col in trend_df.columns]
            
            st.dataframe(
                trend_df[available_trend_columns],
                use_container_width=True
            )
        else:
            st.info("No significant trends detected")
    
    with tab4:
        st.subheader("Export Data")
        
        if records:
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # CSV export
            csv_data = df.to_csv(index=False)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_etl_output_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary"
            )
            
            # JSON export
            json_data = json.dumps(records, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=filename.replace('.csv', '.json'),
                mime="application/json"
            )
            
            # Show export info
            st.info(f"📊 Export includes {len(records)} records with enhanced visual data")
            
            # Sample of what's included
            with st.expander("📋 Export Preview"):
                st.json(records[0] if records else {})
        else:
            st.info("No data available for export")

if __name__ == "__main__":
    main()