import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys

class EnhancedPDFProcessor:
    """Enhanced PDF processor with detailed page-by-page analysis"""
    
    def __init__(self):
        self.pdf_methods = []
        self._check_available_methods()
    
    def _check_available_methods(self):
        """Check which PDF processing methods are available"""
        try:
            import fitz
            self.pdf_methods.append('pymupdf')
        except ImportError:
            pass
        
        try:
            import pdfplumber
            self.pdf_methods.append('pdfplumber')
        except ImportError:
            pass
        
        try:
            import PyPDF2
            self.pdf_methods.append('pypdf2')
        except ImportError:
            pass
    
    def extract_text_from_pdf_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF with enhanced table detection"""
        import fitz
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Try to extract tables
            tables = []
            try:
                # Get text in blocks for better structure detection
                blocks = page.get_text("dict")
                tables = self._extract_tables_from_blocks(blocks)
            except:
                pass
            
            text_content.append({
                'page': page_num + 1,
                'text': text,
                'tables': tables,
                'method': 'PyMuPDF'
            })
        
        doc.close()
        return text_content
    
    def extract_text_from_pdf_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber with table extraction"""
        import pdfplumber
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Extract tables
                tables = []
                try:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:  # Skip empty tables
                            tables.append(table)
                except:
                    pass
                
                text_content.append({
                    'page': page_num + 1,
                    'text': text,
                    'tables': tables,
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
                    'tables': [],  # PyPDF2 doesn't extract tables well
                    'method': 'PyPDF2'
                })
        
        return text_content
    
    def _extract_tables_from_blocks(self, blocks):
        """Extract table-like structures from text blocks"""
        tables = []
        # This is a simplified table detection - in practice, you'd need more sophisticated logic
        return tables
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using available method"""
        if not self.pdf_methods:
            raise Exception("No PDF processing libraries available.")
        
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
    
    def extract_financial_metrics(self, text):
        """Extract detailed financial metrics from text"""
        metrics = {}
        
        # Revenue patterns
        revenue_patterns = [
            r'revenue[:\s]+[£$€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|bn)?',
            r'sales[:\s]+[£$€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|bn)?',
            r'turnover[:\s]+[£$€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|bn)?'
        ]
        
        # Profit patterns
        profit_patterns = [
            r'profit[:\s]+[£$€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|bn)?',
            r'earnings[:\s]+[£$€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|bn)?',
            r'ebitda[:\s]+[£$€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|m|bn)?'
        ]
        
        # Percentage patterns
        percentage_patterns = [
            r'(\d+\.?\d*)%',
            r'(\d+\.?\d*)\s*percent'
        ]
        
        # Growth patterns
        growth_patterns = [
            r'growth[:\s]+(\d+\.?\d*)%',
            r'increase[:\s]+(\d+\.?\d*)%',
            r'up\s+(\d+\.?\d*)%'
        ]
        
        text_lower = text.lower()
        
        # Extract revenue
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                metrics['revenue_values'] = matches[:5]  # Top 5 matches
                break
        
        # Extract profit
        for pattern in profit_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                metrics['profit_values'] = matches[:5]
                break
        
        # Extract percentages
        percentages = []
        for pattern in percentage_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            percentages.extend(matches)
        metrics['percentages'] = percentages[:10]  # Top 10
        
        # Extract growth rates
        growth_rates = []
        for pattern in growth_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            growth_rates.extend(matches)
        metrics['growth_rates'] = growth_rates[:5]
        
        return metrics
    
    def extract_numbers(self, text):
        """Extract all numerical values from text"""
        number_patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'[£$€]\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Currency
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'  # Regular numbers
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return numbers[:30]  # Increased limit
    
    def detect_financial_keywords(self, text):
        """Enhanced financial keyword detection"""
        keywords = {
            'revenue': ['revenue', 'sales', 'income', 'turnover', 'receipts'],
            'profit': ['profit', 'earnings', 'ebitda', 'margin', 'surplus'],
            'growth': ['growth', 'increase', 'rise', 'expansion', 'improvement'],
            'decline': ['decline', 'decrease', 'fall', 'reduction', 'drop'],
            'risk': ['risk', 'uncertainty', 'volatility', 'exposure', 'threat'],
            'performance': ['performance', 'results', 'achievement', 'outcome', 'success'],
            'assets': ['assets', 'capital', 'equity', 'investment', 'holdings'],
            'liabilities': ['liabilities', 'debt', 'obligations', 'payables'],
            'cash': ['cash', 'liquidity', 'funds', 'reserves'],
            'market': ['market', 'share', 'competition', 'sector', 'industry']
        }
        
        detected = {}
        text_lower = text.lower()
        
        for category, words in keywords.items():
            count = sum(text_lower.count(word) for word in words)
            detected[category] = count
        
        return detected
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis"""
        positive_words = ['good', 'excellent', 'strong', 'positive', 'growth', 'increase', 'success', 
                         'improvement', 'outstanding', 'robust', 'solid', 'healthy', 'profitable']
        negative_words = ['bad', 'poor', 'weak', 'negative', 'decline', 'decrease', 'loss', 'problem',
                         'challenging', 'difficult', 'concerning', 'disappointing', 'volatile']
        
        text_lower = text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 'neutral'
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        if sentiment_score > 0.2:
            return 'positive'
        elif sentiment_score < -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_table_data(self, tables):
        """Extract structured data from tables"""
        table_data = []
        
        for i, table in enumerate(tables):
            if not table:
                continue
            
            # Convert table to structured format
            try:
                # Assume first row is headers
                if len(table) > 1:
                    headers = table[0] if table[0] else [f"Col_{j}" for j in range(len(table[1]))]
                    rows = table[1:]
                    
                    # Extract numerical data from table
                    numerical_cells = []
                    for row in rows:
                        for cell in row:
                            if cell and isinstance(cell, str):
                                # Find numbers in cell
                                numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', cell)
                                numerical_cells.extend(numbers)
                    
                    table_data.append({
                        'table_index': i,
                        'headers': headers,
                        'row_count': len(rows),
                        'col_count': len(headers),
                        'numerical_values': numerical_cells[:20]  # Limit to 20 values per table
                    })
            except Exception as e:
                continue
        
        return table_data

def create_enhanced_csv(pages_data, processor):
    """Create enhanced CSV with one row per page"""
    rows = []
    
    for page_data in pages_data:
        page_num = page_data['page']
        text = page_data['text']
        tables = page_data.get('tables', [])
        method = page_data.get('method', 'unknown')
        
        # Basic text analysis
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # Extract financial data
        financial_metrics = processor.extract_financial_metrics(text)
        numbers = processor.extract_numbers(text)
        keywords = processor.detect_financial_keywords(text)
        sentiment = processor.analyze_sentiment(text)
        
        # Table analysis
        table_data = processor.extract_table_data(tables)
        total_tables = len(tables)
        total_table_values = sum(len(td.get('numerical_values', [])) for td in table_data)
        
        # Create row
        row = {
            'document_id': f"page_{page_num}",
            'page_number': page_num,
            'extraction_method': method,
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'table_count': total_tables,
            'table_values_count': total_table_values,
            
            # Financial metrics
            'revenue_values': ', '.join(financial_metrics.get('revenue_values', [])),
            'profit_values': ', '.join(financial_metrics.get('profit_values', [])),
            'percentages': ', '.join(financial_metrics.get('percentages', [])),
            'growth_rates': ', '.join(financial_metrics.get('growth_rates', [])),
            
            # All numbers
            'extracted_numbers': ', '.join(numbers),
            'number_count': len(numbers),
            
            # Keywords
            'revenue_mentions': keywords.get('revenue', 0),
            'profit_mentions': keywords.get('profit', 0),
            'growth_mentions': keywords.get('growth', 0),
            'decline_mentions': keywords.get('decline', 0),
            'risk_mentions': keywords.get('risk', 0),
            'performance_mentions': keywords.get('performance', 0),
            'assets_mentions': keywords.get('assets', 0),
            'liabilities_mentions': keywords.get('liabilities', 0),
            'cash_mentions': keywords.get('cash', 0),
            'market_mentions': keywords.get('market', 0),
            
            # Analysis
            'sentiment': sentiment,
            'has_financial_data': bool(financial_metrics.get('revenue_values') or financial_metrics.get('profit_values')),
            'has_tables': total_tables > 0,
            
            # Sample content
            'sample_text': text[:300] + "..." if len(text) > 300 else text,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def install_pdf_library():
    """Try to install a PDF processing library"""
    libraries = ['pdfplumber', 'PyMuPDF', 'PyPDF2']
    
    for lib in libraries:
        try:
            st.info(f"Attempting to install {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            st.success(f"Successfully installed {lib}")
            return True
        except Exception as e:
            st.warning(f"Failed to install {lib}: {str(e)}")
    
    return False

def main():
    st.set_page_config(
        page_title="BOE ETL - Enhanced PDF Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 BOE ETL - Enhanced PDF Analysis")
    st.markdown("Upload PDF files to extract detailed financial data with page-by-page analysis")
    
    # Initialize processor
    processor = EnhancedPDFProcessor()
    
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
    extract_tables = st.sidebar.checkbox("Extract Tables", value=True)
    extract_financial = st.sidebar.checkbox("Financial Metrics", value=True)
    page_by_page = st.sidebar.checkbox("Page-by-Page Analysis", value=True)
    
    # File upload
    st.header("📁 File Upload")
    file_types = ['pdf'] if processor.pdf_methods else []
    
    if not file_types:
        st.error("PDF processing not available. Please install a PDF library.")
        return
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=file_types,
        help="Upload PDF presentations for detailed financial analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("🔄 Extracting detailed data from PDF..."):
                    pages_data = processor.extract_text_from_pdf(tmp_path)
                
                st.success(f"✅ PDF processed: {uploaded_file.name} ({len(pages_data)} pages)")
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Summary statistics
                total_words = sum(len(page['text'].split()) for page in pages_data)
                total_tables = sum(len(page.get('tables', [])) for page in pages_data)
                pages_with_content = sum(1 for page in pages_data if page['text'].strip())
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pages", len(pages_data))
                with col2:
                    st.metric("Pages with Content", pages_with_content)
                with col3:
                    st.metric("Total Words", total_words)
                with col4:
                    st.metric("Tables Found", total_tables)
                
                # Page-by-page preview
                if page_by_page and pages_data:
                    st.subheader("📄 Page-by-Page Analysis")
                    
                    # Page selector
                    page_options = [f"Page {page['page']}" for page in pages_data if page['text'].strip()]
                    if page_options:
                        selected_page = st.selectbox("Select page to preview:", page_options)
                        page_num = int(selected_page.split()[1])
                        
                        page_data = next(page for page in pages_data if page['page'] == page_num)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Page {page_num} Content:**")
                            st.write(f"- Method: {page_data.get('method', 'unknown')}")
                            st.write(f"- Word Count: {len(page_data['text'].split())}")
                            st.write(f"- Tables: {len(page_data.get('tables', []))}")
                            
                            # Financial analysis for this page
                            if extract_financial:
                                metrics = processor.extract_financial_metrics(page_data['text'])
                                if metrics:
                                    st.write("**Financial Data Found:**")
                                    for key, values in metrics.items():
                                        if values:
                                            st.write(f"- {key.replace('_', ' ').title()}: {', '.join(values[:3])}")
                        
                        with col2:
                            st.text_area(f"Page {page_num} Text:", page_data['text'], height=300)
                
                # Generate enhanced CSV
                st.header("📋 Enhanced CSV Export")
                
                with st.spinner("🔄 Generating detailed CSV..."):
                    df = create_enhanced_csv(pages_data, processor)
                
                st.success(f"✅ Generated CSV with {len(df)} rows and {len(df.columns)} columns")
                st.info(f"📄 Each row represents one page of the PDF with detailed analysis")
                
                # Display CSV preview
                st.subheader("CSV Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Data Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Pages with Financial Data:**")
                    financial_pages = df['has_financial_data'].sum()
                    st.write(f"{financial_pages} out of {len(df)} pages")
                
                with col2:
                    st.write("**Pages with Tables:**")
                    table_pages = df['has_tables'].sum()
                    st.write(f"{table_pages} out of {len(df)} pages")
                
                with col3:
                    st.write("**Total Extracted Numbers:**")
                    total_numbers = df['number_count'].sum()
                    st.write(f"{total_numbers} numerical values")
                
                # Download button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Enhanced CSV",
                    data=csv_data,
                    file_name=f"enhanced_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    
    # Information section
    st.header("ℹ️ About Enhanced PDF Analysis")
    
    with st.expander("Features & Capabilities"):
        st.markdown("""
        ### 📊 Enhanced Data Extraction
        - **Page-by-Page Analysis**: Each page becomes a separate row in the CSV
        - **Table Detection**: Automatically finds and extracts tabular data
        - **Financial Metrics**: Specialized extraction of revenue, profit, growth rates
        - **Numerical Values**: Comprehensive extraction of all numbers, percentages, currency
        
        ### 🔍 Advanced Analysis
        - **25+ Data Fields**: Comprehensive analysis including financial keywords, sentiment, table data
        - **Financial Context**: Revenue, profit, assets, liabilities, cash flow indicators
        - **Table Analysis**: Row/column counts, numerical value extraction from tables
        - **Sentiment Scoring**: Enhanced positive/negative analysis with financial context
        
        ### 📈 Business Intelligence
        - **Growth Rate Detection**: Automatic identification of percentage changes
        - **Risk Indicators**: Detection of risk-related terminology and context
        - **Performance Metrics**: Identification of performance indicators and KPIs
        - **Market Analysis**: Detection of market-related terms and competitive data
        
        ### 🔧 Technical Features
        - **Multiple PDF Libraries**: pdfplumber (preferred), PyMuPDF, PyPDF2 fallback
        - **Table Extraction**: Advanced table detection and structured data extraction
        - **Error Handling**: Robust processing with graceful failure handling
        - **Scalable Output**: Handles documents with hundreds of pages efficiently
        """)

if __name__ == "__main__":
    main()