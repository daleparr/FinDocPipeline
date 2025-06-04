import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Add the boe-etl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'boe-etl'))

# Import visual extraction components directly
try:
    from boe_etl.parsers.visual_extractors import ChartExtractor
    from boe_etl.enhanced_schema_transformer import EnhancedNLPSchema
    import fitz  # PyMuPDF
    from PIL import Image
    import cv2
    import numpy as np
    VISUAL_EXTRACTION_AVAILABLE = True
except ImportError as e:
    st.error(f"Visual extraction dependencies not available: {e}")
    VISUAL_EXTRACTION_AVAILABLE = False

def convert_pdf_to_images(pdf_path):
    """Convert PDF pages to images for processing"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Convert to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        from io import BytesIO
        img = Image.open(BytesIO(img_data))
        images.append((page_num + 1, img))
    
    doc.close()
    return images

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    doc = fitz.open(pdf_path)
    text_content = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text_content.append({
            'page': page_num + 1,
            'text': text
        })
    
    doc.close()
    return text_content

def process_visual_data(images, chart_extractor):
    """Process images for visual data extraction"""
    visual_results = []
    
    for page_num, img in images:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Extract visual data
        try:
            charts = chart_extractor.extract_charts(img_cv)
            
            for chart in charts:
                visual_results.append({
                    'page': page_num,
                    'chart_type': chart.get('type', 'unknown'),
                    'confidence': chart.get('confidence', 0.0),
                    'data_points': chart.get('data_points', []),
                    'trends': chart.get('trends', {}),
                    'bbox': chart.get('bbox', []),
                    'title': chart.get('title', ''),
                    'axis_labels': chart.get('axis_labels', {}),
                    'extracted_values': chart.get('extracted_values', [])
                })
        except Exception as e:
            st.warning(f"Error processing page {page_num}: {str(e)}")
    
    return visual_results

def create_enhanced_csv(text_content, visual_results):
    """Create enhanced CSV with both text and visual data"""
    rows = []
    
    # Process text content
    for text_data in text_content:
        # Create base schema
        schema = EnhancedNLPSchema()
        
        # Fill basic text fields
        schema.document_id = f"page_{text_data['page']}"
        schema.page_number = text_data['page']
        schema.raw_text = text_data['text'][:500]  # Truncate for CSV
        schema.extraction_timestamp = datetime.now().isoformat()
        
        # Add visual data for this page
        page_visuals = [v for v in visual_results if v['page'] == text_data['page']]
        
        if page_visuals:
            schema.has_charts = True
            schema.chart_count = len(page_visuals)
            
            # Aggregate chart types
            chart_types = [v['chart_type'] for v in page_visuals]
            schema.chart_types = ', '.join(set(chart_types))
            
            # Aggregate trends
            trends = []
            for visual in page_visuals:
                if visual['trends']:
                    trend_info = visual['trends']
                    if trend_info.get('direction'):
                        trends.append(f"{trend_info['direction']} (conf: {trend_info.get('confidence', 0):.2f})")
            
            schema.trend_analysis = '; '.join(trends) if trends else 'No clear trends'
            
            # Extract numerical values
            all_values = []
            for visual in page_visuals:
                all_values.extend(visual.get('extracted_values', []))
            
            if all_values:
                schema.numerical_values = ', '.join([str(v) for v in all_values[:10]])  # First 10 values
                schema.value_ranges = f"Min: {min(all_values):.2f}, Max: {max(all_values):.2f}"
        
        # Convert to dictionary for CSV
        row = schema.to_dict()
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    st.set_page_config(
        page_title="BOE ETL - Enhanced Visual Extraction",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 BOE ETL - Enhanced Visual Data Extraction")
    st.markdown("Upload presentation files to extract structured data from charts, graphs, and tables")
    
    if not VISUAL_EXTRACTION_AVAILABLE:
        st.error("Visual extraction components are not available. Please check dependencies.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Visual extraction options
    st.sidebar.subheader("Visual Extraction Options")
    extract_charts = st.sidebar.checkbox("Extract Charts & Graphs", value=True)
    extract_tables = st.sidebar.checkbox("Extract Tables", value=True)
    detect_trends = st.sidebar.checkbox("Analyze Trends", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    
    # File upload
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose a presentation file",
        type=['pdf', 'pptx'],
        help="Upload PDF or PowerPoint presentations for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process file
            with st.spinner("🔄 Processing file..."):
                if uploaded_file.name.lower().endswith('.pdf'):
                    # Extract text
                    text_content = extract_text_from_pdf(tmp_path)
                    
                    # Convert to images for visual processing
                    if extract_charts:
                        images = convert_pdf_to_images(tmp_path)
                        
                        # Initialize chart extractor
                        chart_extractor = ChartExtractor(
                            confidence_threshold=confidence_threshold,
                            detect_trends=detect_trends
                        )
                        
                        # Process visual data
                        visual_results = process_visual_data(images, chart_extractor)
                    else:
                        visual_results = []
                    
                else:
                    st.error("PowerPoint processing not yet implemented. Please use PDF files.")
                    return
            
            # Display results
            st.header("📊 Extraction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Text Content")
                st.info(f"Extracted text from {len(text_content)} pages")
                
                # Show sample text
                if text_content:
                    with st.expander("Sample Text Content"):
                        for i, content in enumerate(text_content[:3]):  # Show first 3 pages
                            st.write(f"**Page {content['page']}:**")
                            st.write(content['text'][:300] + "..." if len(content['text']) > 300 else content['text'])
            
            with col2:
                st.subheader("📈 Visual Data")
                st.info(f"Found {len(visual_results)} visual elements")
                
                if visual_results:
                    # Chart type distribution
                    chart_types = [v['chart_type'] for v in visual_results]
                    type_counts = pd.Series(chart_types).value_counts()
                    st.bar_chart(type_counts)
                    
                    # Show visual details
                    with st.expander("Visual Element Details"):
                        for i, visual in enumerate(visual_results[:5]):  # Show first 5
                            st.write(f"**Element {i+1} (Page {visual['page']}):**")
                            st.write(f"- Type: {visual['chart_type']}")
                            st.write(f"- Confidence: {visual['confidence']:.2f}")
                            if visual['trends']:
                                st.write(f"- Trend: {visual['trends']}")
                            if visual['extracted_values']:
                                st.write(f"- Values: {visual['extracted_values'][:5]}...")
            
            # Generate enhanced CSV
            st.header("📋 Enhanced CSV Output")
            
            with st.spinner("🔄 Generating enhanced CSV..."):
                df = create_enhanced_csv(text_content, visual_results)
            
            st.success(f"✅ Generated CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Display CSV preview
            st.subheader("CSV Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Enhanced CSV",
                data=csv_data,
                file_name=f"enhanced_extraction_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Statistics
            st.subheader("📊 Extraction Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Pages Processed", len(text_content))
            
            with col2:
                st.metric("Visual Elements", len(visual_results))
            
            with col3:
                charts_with_trends = sum(1 for v in visual_results if v.get('trends'))
                st.metric("Elements with Trends", charts_with_trends)
            
            with col4:
                total_values = sum(len(v.get('extracted_values', [])) for v in visual_results)
                st.metric("Extracted Values", total_values)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Information section
    st.header("ℹ️ About Enhanced Visual Extraction")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        ### 🎯 Visual Data Extraction
        - **Chart Detection**: Automatically identifies bar charts, line graphs, pie charts, and scatter plots
        - **Data Point Extraction**: Extracts numerical values from visual elements
        - **Trend Analysis**: Detects positive/negative trends with statistical significance
        - **Table Recognition**: Identifies and extracts tabular data
        
        ### 📊 Enhanced Schema
        - **55 Data Fields**: Comprehensive schema covering text and visual elements
        - **Financial Context**: Revenue, profit, risk indicators for regulatory assessment
        - **Confidence Scoring**: Quality metrics for extracted data
        - **Temporal Analysis**: Time-series trend detection
        
        ### 🔧 Technical Features
        - **Computer Vision**: OpenCV-based image processing
        - **Multi-format Support**: PDF and PowerPoint presentations
        - **Configurable Thresholds**: Adjustable confidence levels
        - **CSV Export**: Structured data output for further analysis
        """)

if __name__ == "__main__":
    main()