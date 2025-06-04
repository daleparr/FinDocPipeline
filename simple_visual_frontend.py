import streamlit as st
import pandas as pd
import tempfile
import json
from datetime import datetime
import os
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO

# Standalone visual extraction classes (copied to avoid dependency issues)
class SimpleChartExtractor:
    def __init__(self, confidence_threshold=0.5, detect_trends=True):
        self.confidence_threshold = confidence_threshold
        self.detect_trends = detect_trends
    
    def extract_charts(self, image):
        """Extract charts from image using basic computer vision"""
        charts = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Filter by area
                area = cv2.contourArea(contour)
                if area < 1000:  # Skip small contours
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic chart classification based on shape
                aspect_ratio = w / h
                chart_type = self._classify_chart_shape(aspect_ratio, contour)
                
                # Extract basic data
                roi = gray[y:y+h, x:x+w]
                extracted_values = self._extract_basic_values(roi)
                
                chart_data = {
                    'type': chart_type,
                    'confidence': min(0.8, area / 10000),  # Simple confidence based on size
                    'bbox': [x, y, w, h],
                    'data_points': len(extracted_values),
                    'extracted_values': extracted_values,
                    'trends': self._analyze_trends(extracted_values) if self.detect_trends else {},
                    'title': f'Chart_{i+1}',
                    'axis_labels': {'x': 'X-axis', 'y': 'Y-axis'}
                }
                
                if chart_data['confidence'] >= self.confidence_threshold:
                    charts.append(chart_data)
        
        except Exception as e:
            st.warning(f"Error in chart extraction: {str(e)}")
        
        return charts
    
    def _classify_chart_shape(self, aspect_ratio, contour):
        """Basic chart type classification"""
        if aspect_ratio > 1.5:
            return 'bar_chart'
        elif aspect_ratio < 0.7:
            return 'column_chart'
        else:
            # Check if roughly circular (pie chart)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    return 'pie_chart'
            return 'line_chart'
    
    def _extract_basic_values(self, roi):
        """Extract basic numerical patterns from image region"""
        # Simple approach: look for intensity variations that might represent data
        values = []
        try:
            # Analyze horizontal and vertical projections
            h_proj = np.sum(roi, axis=0)
            v_proj = np.sum(roi, axis=1)
            
            # Find peaks in projections (simplified data point detection)
            h_peaks = self._find_simple_peaks(h_proj)
            v_peaks = self._find_simple_peaks(v_proj)
            
            # Generate some representative values
            for i, peak in enumerate(h_peaks[:10]):  # Limit to 10 values
                values.append(float(peak * 0.01))  # Scale down
        
        except Exception:
            pass
        
        return values[:10]  # Return max 10 values
    
    def _find_simple_peaks(self, signal):
        """Simple peak detection"""
        peaks = []
        if len(signal) < 3:
            return peaks
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(signal[i])
        
        return sorted(peaks, reverse=True)[:5]  # Top 5 peaks
    
    def _analyze_trends(self, values):
        """Basic trend analysis"""
        if len(values) < 2:
            return {'direction': 'insufficient_data', 'confidence': 0.0}
        
        # Simple linear trend
        x = list(range(len(values)))
        if len(values) > 1:
            slope = (values[-1] - values[0]) / (len(values) - 1)
            if slope > 0.1:
                return {'direction': 'increasing', 'confidence': 0.7}
            elif slope < -0.1:
                return {'direction': 'decreasing', 'confidence': 0.7}
            else:
                return {'direction': 'stable', 'confidence': 0.6}
        
        return {'direction': 'unknown', 'confidence': 0.0}

class SimpleNLPSchema:
    """Simplified schema for data extraction"""
    def __init__(self):
        # Basic fields
        self.document_id = ""
        self.page_number = 0
        self.raw_text = ""
        self.extraction_timestamp = ""
        
        # Visual fields
        self.has_charts = False
        self.chart_count = 0
        self.chart_types = ""
        self.trend_analysis = ""
        self.numerical_values = ""
        self.value_ranges = ""
        
        # Additional fields
        self.confidence_score = 0.0
        self.processing_notes = ""
    
    def to_dict(self):
        """Convert to dictionary for CSV export"""
        return {
            'document_id': self.document_id,
            'page_number': self.page_number,
            'raw_text': self.raw_text,
            'extraction_timestamp': self.extraction_timestamp,
            'has_charts': self.has_charts,
            'chart_count': self.chart_count,
            'chart_types': self.chart_types,
            'trend_analysis': self.trend_analysis,
            'numerical_values': self.numerical_values,
            'value_ranges': self.value_ranges,
            'confidence_score': self.confidence_score,
            'processing_notes': self.processing_notes
        }

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
        schema = SimpleNLPSchema()
        
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
                schema.numerical_values = ', '.join([f"{v:.2f}" for v in all_values[:10]])  # First 10 values
                schema.value_ranges = f"Min: {min(all_values):.2f}, Max: {max(all_values):.2f}"
            
            # Calculate confidence
            confidences = [v.get('confidence', 0) for v in page_visuals]
            schema.confidence_score = sum(confidences) / len(confidences) if confidences else 0.0
        
        schema.processing_notes = f"Processed {len(page_visuals)} visual elements"
        
        # Convert to dictionary for CSV
        row = schema.to_dict()
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    st.set_page_config(
        page_title="BOE ETL - Simple Visual Extraction",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 BOE ETL - Simple Visual Data Extraction")
    st.markdown("Upload PDF presentation files to extract structured data from charts and graphs")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Visual extraction options
    st.sidebar.subheader("Visual Extraction Options")
    extract_charts = st.sidebar.checkbox("Extract Charts & Graphs", value=True)
    detect_trends = st.sidebar.checkbox("Analyze Trends", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    
    # File upload
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF presentation file",
        type=['pdf'],
        help="Upload PDF presentations for visual data analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process file
            with st.spinner("🔄 Processing file..."):
                # Extract text
                text_content = extract_text_from_pdf(tmp_path)
                
                # Convert to images for visual processing
                visual_results = []
                if extract_charts:
                    images = convert_pdf_to_images(tmp_path)
                    
                    # Initialize chart extractor
                    chart_extractor = SimpleChartExtractor(
                        confidence_threshold=confidence_threshold,
                        detect_trends=detect_trends
                    )
                    
                    # Process visual data
                    visual_results = process_visual_data(images, chart_extractor)
            
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
                                st.write(f"- Values: {[f'{v:.2f}' for v in visual['extracted_values'][:5]]}...")
            
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
                file_name=f"simple_extraction_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
    st.header("ℹ️ About Simple Visual Extraction")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        ### 🎯 Visual Data Extraction
        - **Chart Detection**: Basic identification of charts and graphs using computer vision
        - **Data Point Extraction**: Simple numerical value extraction from visual elements
        - **Trend Analysis**: Basic trend detection (increasing/decreasing/stable)
        - **Shape Classification**: Bar charts, line graphs, pie charts, column charts
        
        ### 📊 Data Schema
        - **12 Data Fields**: Essential fields for text and visual data
        - **Confidence Scoring**: Quality metrics for extracted data
        - **CSV Export**: Structured data output for analysis
        
        ### 🔧 Technical Features
        - **Computer Vision**: OpenCV-based image processing
        - **PDF Support**: Text and image extraction from PDF files
        - **Configurable Thresholds**: Adjustable confidence levels
        - **No External Dependencies**: Self-contained visual extraction
        """)

if __name__ == "__main__":
    main()