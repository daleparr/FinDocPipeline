import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re

class MinimalTextProcessor:
    """Minimal text processor using only built-in Python libraries"""
    
    def __init__(self):
        pass
    
    def process_text_file(self, file_content):
        """Process text content and extract basic information"""
        lines = file_content.split('\n')
        
        analysis = {
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'total_words': len(file_content.split()),
            'total_chars': len(file_content),
            'content': file_content
        }
        
        return analysis
    
    def extract_numbers(self, text):
        """Extract numerical values from text"""
        # Find numbers (including decimals, percentages, currency)
        number_patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'[£$€]\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Currency
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'  # Regular numbers
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return numbers[:20]  # Limit to first 20 numbers
    
    def detect_financial_keywords(self, text):
        """Detect financial and business keywords"""
        keywords = {
            'revenue': ['revenue', 'sales', 'income', 'turnover'],
            'profit': ['profit', 'earnings', 'ebitda', 'margin'],
            'growth': ['growth', 'increase', 'rise', 'expansion'],
            'decline': ['decline', 'decrease', 'fall', 'reduction'],
            'risk': ['risk', 'uncertainty', 'volatility', 'exposure'],
            'performance': ['performance', 'results', 'achievement', 'outcome']
        }
        
        detected = {}
        text_lower = text.lower()
        
        for category, words in keywords.items():
            count = sum(text_lower.count(word) for word in words)
            detected[category] = count
        
        return detected
    
    def analyze_sentiment(self, text):
        """Basic sentiment analysis using keyword counting"""
        positive_words = ['good', 'excellent', 'strong', 'positive', 'growth', 'increase', 'success', 'improvement']
        negative_words = ['bad', 'poor', 'weak', 'negative', 'decline', 'decrease', 'loss', 'problem']
        
        text_lower = text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

def create_analysis_csv(analysis, processor):
    """Create CSV with text analysis"""
    
    # Extract numbers and keywords
    numbers = processor.extract_numbers(analysis['content'])
    keywords = processor.detect_financial_keywords(analysis['content'])
    sentiment = processor.analyze_sentiment(analysis['content'])
    
    row = {
        'document_id': 'uploaded_document',
        'total_lines': analysis['total_lines'],
        'non_empty_lines': analysis['non_empty_lines'],
        'total_words': analysis['total_words'],
        'total_chars': analysis['total_chars'],
        'extracted_numbers': ', '.join(numbers),
        'number_count': len(numbers),
        'revenue_mentions': keywords.get('revenue', 0),
        'profit_mentions': keywords.get('profit', 0),
        'growth_mentions': keywords.get('growth', 0),
        'decline_mentions': keywords.get('decline', 0),
        'risk_mentions': keywords.get('risk', 0),
        'performance_mentions': keywords.get('performance', 0),
        'sentiment': sentiment,
        'sample_text': analysis['content'][:500] + "..." if len(analysis['content']) > 500 else analysis['content'],
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    return pd.DataFrame([row])

def main():
    st.set_page_config(
        page_title="BOE ETL - Minimal Text Analysis",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 BOE ETL - Minimal Text Analysis")
    st.markdown("Upload text files to extract and analyze content")
    
    # Sidebar
    st.sidebar.header("⚙️ Analysis Options")
    include_numbers = st.sidebar.checkbox("Extract Numbers", value=True)
    include_keywords = st.sidebar.checkbox("Financial Keywords", value=True)
    include_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    
    # File upload
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt', 'csv', 'md'],
        help="Upload text files for analysis"
    )
    
    # Alternative: Text input
    st.subheader("Or paste text directly:")
    text_input = st.text_area("Paste your text here:", height=200)
    
    # Process file or text input
    content_to_process = None
    source_name = None
    
    if uploaded_file is not None:
        # Read uploaded file
        try:
            content_to_process = uploaded_file.read().decode('utf-8')
            source_name = uploaded_file.name
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    elif text_input.strip():
        content_to_process = text_input
        source_name = "pasted_text"
        st.success("✅ Text input received")
    
    if content_to_process:
        # Process content
        with st.spinner("🔄 Processing content..."):
            processor = MinimalTextProcessor()
            analysis = processor.process_text_file(content_to_process)
        
        # Display results
        st.header("📊 Analysis Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Lines", analysis['total_lines'])
        with col2:
            st.metric("Non-empty Lines", analysis['non_empty_lines'])
        with col3:
            st.metric("Total Words", analysis['total_words'])
        with col4:
            st.metric("Total Characters", analysis['total_chars'])
        
        # Content preview
        st.subheader("📄 Content Preview")
        with st.expander("View Content", expanded=False):
            st.text_area("Content:", analysis['content'], height=300)
        
        # Analysis features
        if include_numbers or include_keywords or include_sentiment:
            st.subheader("🔍 Content Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if include_numbers:
                    numbers = processor.extract_numbers(analysis['content'])
                    st.write("**Extracted Numbers:**")
                    if numbers:
                        for num in numbers[:10]:
                            st.write(f"- {num}")
                    else:
                        st.write("No numbers found")
                
                if include_sentiment:
                    sentiment = processor.analyze_sentiment(analysis['content'])
                    st.write(f"**Overall Sentiment:** {sentiment.title()}")
            
            with col2:
                if include_keywords:
                    keywords = processor.detect_financial_keywords(analysis['content'])
                    st.write("**Financial Keywords:**")
                    for category, count in keywords.items():
                        if count > 0:
                            st.write(f"- {category.title()}: {count}")
        
        # Generate CSV
        st.header("📋 CSV Export")
        
        with st.spinner("🔄 Generating CSV..."):
            df = create_analysis_csv(analysis, processor)
        
        st.success(f"✅ Generated CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Display CSV preview
        st.subheader("CSV Preview")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis CSV",
            data=csv_data,
            file_name=f"text_analysis_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Demo section
    if not content_to_process:
        st.header("🎯 Demo")
        if st.button("Load Demo Text"):
            demo_text = """
            Financial Performance Report Q4 2024
            
            Revenue increased by 15% to £2.5 billion, exceeding expectations.
            Profit margins improved to 12.3%, up from 10.8% in Q3.
            
            Key highlights:
            - Strong growth in digital services (+25%)
            - Cost reduction initiatives saved £50 million
            - Risk exposure decreased by 8%
            
            Outlook: We expect continued positive performance in 2025,
            with revenue growth projected at 10-12%.
            
            Challenges include market volatility and regulatory uncertainty.
            """
            
            st.text_area("Demo Text (you can edit this):", demo_text, height=200, key="demo_text")
            st.info("👆 Copy this text to the input area above to try the analysis")
    
    # Information section
    st.header("ℹ️ About Minimal Text Analysis")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        ### 📝 Text Processing
        - **File Upload**: Support for TXT, CSV, MD files
        - **Direct Input**: Paste text directly for analysis
        - **Basic Statistics**: Line count, word count, character count
        
        ### 🔍 Content Analysis
        - **Number Extraction**: Find numerical values, percentages, currency amounts
        - **Financial Keywords**: Detect revenue, profit, growth, risk-related terms
        - **Sentiment Analysis**: Basic positive/negative/neutral classification
        
        ### 📊 Data Export
        - **CSV Format**: Structured data export for further analysis
        - **Comprehensive Fields**: 15+ data fields covering all analysis aspects
        - **Timestamp Tracking**: Processing time information
        
        ### 🔧 Technical Features
        - **Built-in Libraries Only**: No external dependencies
        - **Regex Pattern Matching**: Advanced number and keyword detection
        - **Cross-platform Compatible**: Works on any system with Python
        """)

if __name__ == "__main__":
    main()