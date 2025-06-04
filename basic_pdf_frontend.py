import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import fitz  # PyMuPDF
import re

class BasicPDFProcessor:
    """Basic PDF processor for text extraction and simple analysis"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with page information"""
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Basic text analysis
            word_count = len(text.split())
            char_count = len(text)
            
            text_content.append({
                'page': page_num + 1,
                'text': text,
                'word_count': word_count,
                'char_count': char_count,
                'has_content': bool(text.strip())
            })
        
        doc.close()
        return text_content
    
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

def create_analysis_csv(text_content, processor):
    """Create CSV with text analysis"""
    rows = []
    
    for page_data in text_content:
        # Extract numbers and keywords
        numbers = processor.extract_numbers(page_data['text'])
        keywords = processor.detect_financial_keywords(page_data['text'])
        sentiment = processor.analyze_sentiment(page_data['text'])
        
        row = {
            'document_id': f"page_{page_data['page']}",
            'page_number': page_data['page'],
            'word_count': page_data['word_count'],
            'char_count': page_data['char_count'],
            'has_content': page_data['has_content'],
            'extracted_numbers': ', '.join(numbers),
            'number_count': len(numbers),
            'revenue_mentions': keywords.get('revenue', 0),
            'profit_mentions': keywords.get('profit', 0),
            'growth_mentions': keywords.get('growth', 0),
            'decline_mentions': keywords.get('decline', 0),
            'risk_mentions': keywords.get('risk', 0),
            'performance_mentions': keywords.get('performance', 0),
            'sentiment': sentiment,
            'sample_text': page_data['text'][:200] + "..." if len(page_data['text']) > 200 else page_data['text'],
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    st.set_page_config(
        page_title="BOE ETL - Basic PDF Analysis",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 BOE ETL - Basic PDF Text Analysis")
    st.markdown("Upload PDF presentation files to extract and analyze text content")
    
    # Sidebar
    st.sidebar.header("⚙️ Analysis Options")
    include_numbers = st.sidebar.checkbox("Extract Numbers", value=True)
    include_keywords = st.sidebar.checkbox("Financial Keywords", value=True)
    include_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    
    # File upload
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload PDF presentations for text analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process file
            with st.spinner("🔄 Processing PDF..."):
                processor = BasicPDFProcessor()
                text_content = processor.extract_text_from_pdf(tmp_path)
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Summary statistics
            total_pages = len(text_content)
            total_words = sum(page['word_count'] for page in text_content)
            pages_with_content = sum(1 for page in text_content if page['has_content'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pages", total_pages)
            with col2:
                st.metric("Total Words", total_words)
            with col3:
                st.metric("Pages with Content", pages_with_content)
            with col4:
                st.metric("Avg Words/Page", int(total_words / total_pages) if total_pages > 0 else 0)
            
            # Text content preview
            st.subheader("📄 Text Content Preview")
            
            if text_content:
                # Show page selector
                page_options = [f"Page {page['page']}" for page in text_content if page['has_content']]
                if page_options:
                    selected_page = st.selectbox("Select page to preview:", page_options)
                    page_num = int(selected_page.split()[1])
                    
                    # Find the selected page data
                    page_data = next(page for page in text_content if page['page'] == page_num)
                    
                    with st.expander(f"Page {page_num} Content", expanded=True):
                        st.write(f"**Word Count:** {page_data['word_count']}")
                        st.write(f"**Character Count:** {page_data['char_count']}")
                        st.text_area("Text Content:", page_data['text'], height=200)
            
            # Analysis features
            if include_numbers or include_keywords or include_sentiment:
                st.subheader("🔍 Content Analysis")
                
                analysis_data = []
                for page_data in text_content:
                    if not page_data['has_content']:
                        continue
                    
                    analysis = {'page': page_data['page']}
                    
                    if include_numbers:
                        numbers = processor.extract_numbers(page_data['text'])
                        analysis['numbers'] = numbers
                    
                    if include_keywords:
                        keywords = processor.detect_financial_keywords(page_data['text'])
                        analysis['keywords'] = keywords
                    
                    if include_sentiment:
                        sentiment = processor.analyze_sentiment(page_data['text'])
                        analysis['sentiment'] = sentiment
                    
                    analysis_data.append(analysis)
                
                # Display analysis
                if analysis_data:
                    for analysis in analysis_data[:5]:  # Show first 5 pages
                        with st.expander(f"Page {analysis['page']} Analysis"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if 'numbers' in analysis:
                                    st.write("**Extracted Numbers:**")
                                    if analysis['numbers']:
                                        for num in analysis['numbers'][:10]:
                                            st.write(f"- {num}")
                                    else:
                                        st.write("No numbers found")
                                
                                if 'sentiment' in analysis:
                                    st.write(f"**Sentiment:** {analysis['sentiment'].title()}")
                            
                            with col2:
                                if 'keywords' in analysis:
                                    st.write("**Financial Keywords:**")
                                    for category, count in analysis['keywords'].items():
                                        if count > 0:
                                            st.write(f"- {category.title()}: {count}")
            
            # Generate CSV
            st.header("📋 CSV Export")
            
            with st.spinner("🔄 Generating CSV..."):
                df = create_analysis_csv(text_content, processor)
            
            st.success(f"✅ Generated CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Display CSV preview
            st.subheader("CSV Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis CSV",
                data=csv_data,
                file_name=f"pdf_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Information section
    st.header("ℹ️ About Basic PDF Analysis")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        ### 📄 Text Extraction
        - **PDF Processing**: Extract text content from all pages
        - **Word & Character Counting**: Basic text statistics
        - **Content Detection**: Identify pages with meaningful content
        
        ### 🔍 Content Analysis
        - **Number Extraction**: Find numerical values, percentages, currency amounts
        - **Financial Keywords**: Detect revenue, profit, growth, risk-related terms
        - **Sentiment Analysis**: Basic positive/negative/neutral classification
        
        ### 📊 Data Export
        - **CSV Format**: Structured data export for further analysis
        - **Page-by-Page Analysis**: Detailed breakdown by document page
        - **Timestamp Tracking**: Processing time information
        
        ### 🔧 Technical Features
        - **PyMuPDF Processing**: Reliable PDF text extraction
        - **Regex Pattern Matching**: Advanced number and keyword detection
        - **No External Dependencies**: Uses only standard libraries
        """)

if __name__ == "__main__":
    main()