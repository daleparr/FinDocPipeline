import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys

class PDFTextProcessor:
    """PDF text processor with fallback methods"""
    
    def __init__(self):
        self.pdf_methods = []
        self._check_available_methods()
    
    def _check_available_methods(self):
        """Check which PDF processing methods are available"""
        # Try PyMuPDF first
        try:
            import fitz
            self.pdf_methods.append('pymupdf')
        except ImportError:
            pass
        
        # Try pdfplumber
        try:
            import pdfplumber
            self.pdf_methods.append('pdfplumber')
        except ImportError:
            pass
        
        # Try PyPDF2
        try:
            import PyPDF2
            self.pdf_methods.append('pypdf2')
        except ImportError:
            pass
    
    def extract_text_from_pdf_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF"""
        import fitz
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_content.append({
                'page': page_num + 1,
                'text': text,
                'method': 'PyMuPDF'
            })
        
        doc.close()
        return text_content
    
    def extract_text_from_pdf_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber"""
        import pdfplumber
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text_content.append({
                    'page': page_num + 1,
                    'text': text,
                    'method': 'pdfplumber'
                })
        
        return text_content
    
    def extract_text_from_pdf_pypdf2(self, pdf_path):
        """Extract text using PyPDF2"""
        import PyPDF2
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_content.append({
                    'page': page_num + 1,
                    'text': text,
                    'method': 'PyPDF2'
                })
        
        return text_content
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using available method"""
        if not self.pdf_methods:
            raise Exception("No PDF processing libraries available. Please install PyMuPDF, pdfplumber, or PyPDF2.")
        
        # Try methods in order of preference
        for method in self.pdf_methods:
            try:
                if method == 'pymupdf':
                    return self.extract_text_from_pdf_pymupdf(pdf_path)
                elif method == 'pdfplumber':
                    return self.extract_text_from_pdf_pdfplumber(pdf_path)
                elif method == 'pypdf2':
                    return self.extract_text_from_pdf_pypdf2(pdf_path)
            except Exception as e:
                st.warning(f"Failed to extract with {method}: {str(e)}")
                continue
        
        raise Exception("All PDF extraction methods failed")
    
    def process_text_content(self, text_content):
        """Process extracted text content"""
        if isinstance(text_content, str):
            # Single text string
            analysis = {
                'total_pages': 1,
                'total_lines': len(text_content.split('\n')),
                'non_empty_lines': len([line for line in text_content.split('\n') if line.strip()]),
                'total_words': len(text_content.split()),
                'total_chars': len(text_content),
                'content': text_content,
                'method': 'direct_text'
            }
        else:
            # List of page content
            all_text = '\n'.join([page['text'] for page in text_content])
            analysis = {
                'total_pages': len(text_content),
                'total_lines': len(all_text.split('\n')),
                'non_empty_lines': len([line for line in all_text.split('\n') if line.strip()]),
                'total_words': len(all_text.split()),
                'total_chars': len(all_text),
                'content': all_text,
                'pages': text_content,
                'method': text_content[0]['method'] if text_content else 'unknown'
            }
        
        return analysis
    
    def extract_numbers(self, text):
        """Extract numerical values from text"""
        number_patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'[£$€]\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Currency
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'  # Regular numbers
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return numbers[:20]
    
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
        """Basic sentiment analysis"""
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

def install_pdf_library():
    """Try to install a PDF processing library"""
    libraries = ['PyMuPDF', 'pdfplumber', 'PyPDF2']
    
    for lib in libraries:
        try:
            st.info(f"Attempting to install {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            st.success(f"Successfully installed {lib}")
            return True
        except Exception as e:
            st.warning(f"Failed to install {lib}: {str(e)}")
    
    return False

def create_analysis_csv(analysis, processor):
    """Create CSV with analysis results"""
    numbers = processor.extract_numbers(analysis['content'])
    keywords = processor.detect_financial_keywords(analysis['content'])
    sentiment = processor.analyze_sentiment(analysis['content'])
    
    row = {
        'document_id': 'uploaded_document',
        'extraction_method': analysis.get('method', 'unknown'),
        'total_pages': analysis.get('total_pages', 1),
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
        page_title="BOE ETL - PDF Text Analysis",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 BOE ETL - PDF Text Analysis")
    st.markdown("Upload PDF or text files to extract and analyze content")
    
    # Initialize processor
    processor = PDFTextProcessor()
    
    # Check PDF processing capabilities
    if not processor.pdf_methods:
        st.warning("⚠️ No PDF processing libraries detected!")
        st.info("PDF processing requires one of: PyMuPDF, pdfplumber, or PyPDF2")
        
        if st.button("🔧 Try to install PDF processing library"):
            if install_pdf_library():
                st.experimental_rerun()
            else:
                st.error("Failed to install PDF processing libraries. Please install manually.")
    else:
        st.success(f"✅ PDF processing available using: {', '.join(processor.pdf_methods)}")
    
    # Sidebar
    st.sidebar.header("⚙️ Analysis Options")
    include_numbers = st.sidebar.checkbox("Extract Numbers", value=True)
    include_keywords = st.sidebar.checkbox("Financial Keywords", value=True)
    include_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    
    # File upload
    st.header("📁 File Upload")
    file_types = ['pdf', 'txt', 'csv', 'md'] if processor.pdf_methods else ['txt', 'csv', 'md']
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=file_types,
        help=f"Upload files for analysis. Supported: {', '.join(file_types)}"
    )
    
    # Alternative: Text input
    st.subheader("Or paste text directly:")
    text_input = st.text_area("Paste your text here:", height=200)
    
    # Process file or text input
    content_to_process = None
    source_name = None
    
    if uploaded_file is not None:
        source_name = uploaded_file.name
        
        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                if not processor.pdf_methods:
                    st.error("PDF processing not available. Please install a PDF library or use text files.")
                    return
                
                # Save PDF temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner("🔄 Extracting text from PDF..."):
                        text_content = processor.extract_text_from_pdf(tmp_path)
                        analysis = processor.process_text_content(text_content)
                    
                    st.success(f"✅ PDF processed: {uploaded_file.name}")
                    content_to_process = analysis
                
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            else:
                # Text file
                content_text = uploaded_file.read().decode('utf-8')
                analysis = processor.process_text_content(content_text)
                content_to_process = analysis
                st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    elif text_input.strip():
        analysis = processor.process_text_content(text_input)
        content_to_process = analysis
        source_name = "pasted_text"
        st.success("✅ Text input received")
    
    if content_to_process:
        # Display results
        st.header("📊 Analysis Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pages", content_to_process.get('total_pages', 1))
        with col2:
            st.metric("Total Words", content_to_process['total_words'])
        with col3:
            st.metric("Non-empty Lines", content_to_process['non_empty_lines'])
        with col4:
            st.metric("Extraction Method", content_to_process.get('method', 'unknown'))
        
        # Page-by-page view for PDFs
        if 'pages' in content_to_process and len(content_to_process['pages']) > 1:
            st.subheader("📄 Page-by-Page Content")
            page_options = [f"Page {page['page']}" for page in content_to_process['pages']]
            selected_page = st.selectbox("Select page to preview:", page_options)
            page_num = int(selected_page.split()[1])
            
            page_data = next(page for page in content_to_process['pages'] if page['page'] == page_num)
            
            with st.expander(f"Page {page_num} Content", expanded=True):
                st.write(f"**Method:** {page_data.get('method', 'unknown')}")
                st.write(f"**Word Count:** {len(page_data['text'].split())}")
                st.text_area("Text Content:", page_data['text'], height=200)
        
        else:
            # Single content view
            st.subheader("📄 Content Preview")
            with st.expander("View Content", expanded=False):
                st.text_area("Content:", content_to_process['content'], height=300)
        
        # Analysis features
        if include_numbers or include_keywords or include_sentiment:
            st.subheader("🔍 Content Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if include_numbers:
                    numbers = processor.extract_numbers(content_to_process['content'])
                    st.write("**Extracted Numbers:**")
                    if numbers:
                        for num in numbers[:10]:
                            st.write(f"- {num}")
                    else:
                        st.write("No numbers found")
                
                if include_sentiment:
                    sentiment = processor.analyze_sentiment(content_to_process['content'])
                    st.write(f"**Overall Sentiment:** {sentiment.title()}")
            
            with col2:
                if include_keywords:
                    keywords = processor.detect_financial_keywords(content_to_process['content'])
                    st.write("**Financial Keywords:**")
                    for category, count in keywords.items():
                        if count > 0:
                            st.write(f"- {category.title()}: {count}")
        
        # Generate CSV
        st.header("📋 CSV Export")
        
        with st.spinner("🔄 Generating CSV..."):
            df = create_analysis_csv(content_to_process, processor)
        
        st.success(f"✅ Generated CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Display CSV preview
        st.subheader("CSV Preview")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis CSV",
            data=csv_data,
            file_name=f"analysis_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Information section
    st.header("ℹ️ About PDF Text Analysis")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        ### 📄 File Processing
        - **PDF Support**: Extract text from PDF files using multiple methods
        - **Text Files**: Support for TXT, CSV, MD files
        - **Direct Input**: Paste text directly for analysis
        - **Fallback Methods**: Tries multiple PDF libraries for best compatibility
        
        ### 🔍 Content Analysis
        - **Number Extraction**: Find numerical values, percentages, currency amounts
        - **Financial Keywords**: Detect revenue, profit, growth, risk-related terms
        - **Sentiment Analysis**: Basic positive/negative/neutral classification
        - **Page-by-Page Analysis**: Detailed breakdown for multi-page documents
        
        ### 📊 Data Export
        - **CSV Format**: Structured data export for further analysis
        - **Comprehensive Fields**: 16+ data fields covering all analysis aspects
        - **Method Tracking**: Records which extraction method was used
        
        ### 🔧 Technical Features
        - **Multiple PDF Libraries**: PyMuPDF, pdfplumber, PyPDF2 support
        - **Auto-installation**: Attempts to install missing dependencies
        - **Error Handling**: Graceful fallback between methods
        """)

if __name__ == "__main__":
    main()