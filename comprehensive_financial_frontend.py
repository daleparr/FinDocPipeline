import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys

class ComprehensiveFinancialParser:
    """Comprehensive parser that captures ALL text plus enhanced table/chart interpretation"""
    
    def __init__(self):
        self.pdf_methods = []
        self._check_available_methods()
    
    def _check_available_methods(self):
        """Check which PDF processing methods are available"""
        try:
            import pdfplumber
            self.pdf_methods.append('pdfplumber')
        except ImportError:
            pass
        
        try:
            import fitz
            self.pdf_methods.append('pymupdf')
        except ImportError:
            pass
    
    def extract_comprehensive_data(self, pdf_path):
        """Extract ALL text content plus enhanced structural analysis"""
        if 'pdfplumber' in self.pdf_methods:
            return self._extract_with_pdfplumber(pdf_path)
        elif 'pymupdf' in self.pdf_methods:
            return self._extract_with_pymupdf(pdf_path)
        else:
            raise Exception("No PDF processing libraries available")
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Comprehensive extraction using pdfplumber"""
        import pdfplumber
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract ALL text content
                full_text = page.extract_text() or ""
                
                page_data = {
                    'page': page_num + 1,
                    'method': 'pdfplumber_comprehensive',
                    'full_text': full_text,
                    'word_count': len(full_text.split()),
                    'char_count': len(full_text),
                    'line_count': len(full_text.split('\n')),
                    'tables': [],
                    'table_analysis': {},
                    'financial_metrics': {},
                    'chart_indicators': [],
                    'structured_elements': []
                }
                
                # Extract tables with full structure
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_analysis = self._analyze_table_comprehensive(table, table_idx)
                                page_data['tables'].append(table_analysis)
                                
                                # Add table content to structured elements
                                page_data['structured_elements'].append({
                                    'type': 'table',
                                    'id': table_idx,
                                    'content': table_analysis
                                })
                except Exception as e:
                    st.warning(f"Table extraction error on page {page_num + 1}: {str(e)}")
                
                # Enhanced financial analysis
                page_data['financial_metrics'] = self._extract_comprehensive_financial_data(full_text)
                
                # Chart and visual element detection
                page_data['chart_indicators'] = self._detect_chart_elements(full_text)
                
                # Extract all numerical relationships
                page_data['numerical_analysis'] = self._extract_all_numerical_relationships(full_text)
                
                pages_data.append(page_data)
        
        return pages_data
    
    def _extract_with_pymupdf(self, pdf_path):
        """Comprehensive extraction using PyMuPDF"""
        import fitz
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract ALL text content
            full_text = page.get_text()
            
            page_data = {
                'page': page_num + 1,
                'method': 'pymupdf_comprehensive',
                'full_text': full_text,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'line_count': len(full_text.split('\n')),
                'tables': [],
                'table_analysis': {},
                'financial_metrics': {},
                'chart_indicators': [],
                'structured_elements': []
            }
            
            # Enhanced financial analysis
            page_data['financial_metrics'] = self._extract_comprehensive_financial_data(full_text)
            
            # Chart and visual element detection
            page_data['chart_indicators'] = self._detect_chart_elements(full_text)
            
            # Extract all numerical relationships
            page_data['numerical_analysis'] = self._extract_all_numerical_relationships(full_text)
            
            pages_data.append(page_data)
        
        doc.close()
        return pages_data
    
    def _analyze_table_comprehensive(self, table, table_idx):
        """Comprehensive table analysis preserving all structure"""
        if not table or len(table) == 0:
            return None
        
        # Clean table data
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        # Identify structure
        headers = cleaned_table[0] if cleaned_table else []
        data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
        
        # Extract all table content as text
        table_text = "\n".join(["\t".join(row) for row in cleaned_table])
        
        # Analyze table content
        table_analysis = {
            'table_id': table_idx,
            'headers': headers,
            'data_rows': data_rows,
            'row_count': len(data_rows),
            'col_count': len(headers),
            'table_text': table_text,
            'table_type': self._classify_table_type(headers),
            'financial_data': [],
            'numerical_summary': {}
        }
        
        # Extract financial data from table
        for row_idx, row in enumerate(data_rows):
            for col_idx, cell in enumerate(row):
                if cell and self._contains_financial_data(cell):
                    header = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx}"
                    table_analysis['financial_data'].append({
                        'row': row_idx,
                        'column': col_idx,
                        'header': header,
                        'value': cell,
                        'numeric_value': self._extract_numeric_value(cell),
                        'data_type': self._classify_financial_data_type(cell, header)
                    })
        
        # Numerical summary
        all_numbers = []
        for row in data_rows:
            for cell in row:
                numeric_val = self._extract_numeric_value(cell)
                if numeric_val:
                    try:
                        all_numbers.append(float(numeric_val.replace(',', '')))
                    except:
                        pass
        
        if all_numbers:
            table_analysis['numerical_summary'] = {
                'count': len(all_numbers),
                'min': min(all_numbers),
                'max': max(all_numbers),
                'avg': sum(all_numbers) / len(all_numbers)
            }
        
        return table_analysis
    
    def _extract_comprehensive_financial_data(self, text):
        """Extract comprehensive financial metrics with context"""
        financial_data = {
            'capital_metrics': [],
            'balance_sheet_items': [],
            'performance_metrics': [],
            'ratios': [],
            'yoy_changes': [],
            'currency_amounts': [],
            'percentages': [],
            'all_numbers': []
        }
        
        # Capital metrics patterns
        capital_patterns = {
            'cet1_ratio': r'CET1[^:]*ratio[:\s]*(\d+\.?\d*)%?',
            'tier1_ratio': r'Tier\s*1[^:]*ratio[:\s]*(\d+\.?\d*)%?',
            'capital_ratio': r'Capital\s*ratio[:\s]*(\d+\.?\d*)%?',
            'leverage_ratio': r'Leverage\s*ratio[:\s]*(\d+\.?\d*)%?'
        }
        
        # Balance sheet patterns
        balance_sheet_patterns = {
            'total_assets': r'(?:Total\s*)?Assets[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            'deposits': r'Deposits[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            'equity': r'(?:Total\s*)?Equity[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            'cash': r'Cash[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d+)?)'
        }
        
        # Performance patterns
        performance_patterns = {
            'revenue': r'(?:Net\s*)?Revenue[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            'net_income': r'Net\s*Income[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            'roa': r'ROA[:\s]*(\d+\.?\d*)%?',
            'roe': r'ROE[:\s]*(\d+\.?\d*)%?'
        }
        
        # Extract all pattern types
        pattern_groups = {
            'capital_metrics': capital_patterns,
            'balance_sheet_items': balance_sheet_patterns,
            'performance_metrics': performance_patterns
        }
        
        for group_name, patterns in pattern_groups.items():
            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    financial_data[group_name].append({
                        'metric': metric_name,
                        'value': match,
                        'context': f'extracted_from_{group_name}'
                    })
        
        # Extract YoY changes
        yoy_patterns = [
            r'YoY[:\s]*\(?([+-]?\d+\.?\d*)%?\)?',
            r'year[- ]over[- ]year[:\s]*\(?([+-]?\d+\.?\d*)%?\)?'
        ]
        
        for pattern in yoy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            financial_data['yoy_changes'].extend(matches)
        
        # Extract all currency amounts
        currency_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:[BMK]|million|billion)?'
        currency_matches = re.findall(currency_pattern, text)
        financial_data['currency_amounts'] = currency_matches
        
        # Extract all percentages
        percentage_pattern = r'(\d+\.?\d*)%'
        percentage_matches = re.findall(percentage_pattern, text)
        financial_data['percentages'] = percentage_matches
        
        # Extract ALL numbers
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        all_numbers = re.findall(number_pattern, text)
        financial_data['all_numbers'] = all_numbers
        
        return financial_data
    
    def _detect_chart_elements(self, text):
        """Detect chart and visual elements"""
        chart_indicators = []
        
        # Chart type indicators
        chart_types = [
            'YoY Standardized', 'Walk', 'Waterfall', 'Bridge',
            'Stacked', 'Bar Chart', 'Line Chart', 'Pie Chart'
        ]
        
        # Chart components
        chart_components = [
            'bps', 'basis points', 'Management Buffer', 'GSIB Surcharge',
            'Regulatory Minimum', 'Capital Distribution', 'Net Income',
            'Unrealized', 'RWA', 'DTA Impact'
        ]
        
        # Visual indicators
        visual_indicators = [
            'overview', 'metrics', 'ratio', 'breakdown', 'composition',
            'analysis', 'trend', 'performance', 'comparison'
        ]
        
        all_indicators = chart_types + chart_components + visual_indicators
        
        for indicator in all_indicators:
            if indicator.lower() in text.lower():
                chart_indicators.append(indicator)
        
        return list(set(chart_indicators))  # Remove duplicates
    
    def _extract_all_numerical_relationships(self, text):
        """Extract all numerical relationships and context"""
        relationships = []
        
        # Split text into lines for context analysis
        lines = text.split('\n')
        
        for line_idx, line in enumerate(lines):
            # Find numbers in this line
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', line)
            percentages = re.findall(r'\d+\.?\d*%', line)
            currency = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d+)?', line)
            
            if numbers or percentages or currency:
                # Get context (previous and next lines)
                context_start = max(0, line_idx - 1)
                context_end = min(len(lines), line_idx + 2)
                context = ' '.join(lines[context_start:context_end])
                
                relationships.append({
                    'line_number': line_idx + 1,
                    'line_content': line.strip(),
                    'numbers': numbers,
                    'percentages': percentages,
                    'currency': currency,
                    'context': context.strip()
                })
        
        return relationships
    
    def _classify_table_type(self, headers):
        """Classify table type based on headers"""
        header_text = ' '.join(headers).lower()
        
        if any(term in header_text for term in ['capital', 'ratio', 'cet1', 'tier']):
            return 'capital_metrics'
        elif any(term in header_text for term in ['assets', 'liabilities', 'equity']):
            return 'balance_sheet'
        elif any(term in header_text for term in ['revenue', 'income', 'earnings']):
            return 'income_statement'
        elif any(term in header_text for term in ['cash', 'flow', 'liquidity']):
            return 'cash_flow'
        elif any(term in header_text for term in ['risk', 'rwa', 'exposure']):
            return 'risk_metrics'
        else:
            return 'general_financial'
    
    def _contains_financial_data(self, cell):
        """Check if cell contains financial data"""
        if not cell:
            return False
        
        patterns = [
            r'\$\d+',  # Currency
            r'\d+\.?\d*%',  # Percentage
            r'\d{1,3}(?:,\d{3})+',  # Large numbers
            r'\d+\.?\d*[BMK]',  # Numbers with scale
            r'CET1|Tier|ROA|ROE|bps'  # Financial terms
        ]
        
        return any(re.search(pattern, cell, re.IGNORECASE) for pattern in patterns)
    
    def _extract_numeric_value(self, cell):
        """Extract numeric value from cell"""
        if not cell:
            return None
        
        # Remove non-numeric characters except decimal points and commas
        cleaned = re.sub(r'[^\d.,-]', '', str(cell))
        
        # Extract the main number
        number_match = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', cleaned)
        if number_match:
            return number_match.group()
        
        return None
    
    def _classify_financial_data_type(self, cell, header):
        """Classify the type of financial data"""
        cell_lower = str(cell).lower()
        header_lower = str(header).lower()
        
        if '%' in cell or 'ratio' in header_lower:
            return 'percentage'
        elif '$' in cell or 'amount' in header_lower:
            return 'currency'
        elif any(term in header_lower for term in ['capital', 'cet1', 'tier']):
            return 'capital_metric'
        elif any(term in header_lower for term in ['assets', 'liabilities']):
            return 'balance_sheet_item'
        else:
            return 'general_numeric'

def create_comprehensive_csv(pages_data):
    """Create comprehensive CSV with ALL text plus enhanced analysis"""
    rows = []
    
    for page_data in pages_data:
        page_num = page_data['page']
        
        # Create main row with ALL page content
        main_row = {
            'document_id': f"page_{page_num}",
            'page_number': page_num,
            'data_type': 'complete_page',
            'extraction_method': page_data['method'],
            
            # COMPLETE TEXT CONTENT
            'full_page_text': page_data['full_text'],
            'word_count': page_data['word_count'],
            'char_count': page_data['char_count'],
            'line_count': page_data['line_count'],
            
            # TABLE ANALYSIS
            'table_count': len(page_data.get('tables', [])),
            'table_types': '; '.join([t.get('table_type', '') for t in page_data.get('tables', [])]),
            'table_headers': '; '.join([', '.join(t.get('headers', [])) for t in page_data.get('tables', [])]),
            
            # FINANCIAL METRICS
            'capital_metrics_count': len(page_data.get('financial_metrics', {}).get('capital_metrics', [])),
            'balance_sheet_items_count': len(page_data.get('financial_metrics', {}).get('balance_sheet_items', [])),
            'performance_metrics_count': len(page_data.get('financial_metrics', {}).get('performance_metrics', [])),
            
            # ALL EXTRACTED NUMBERS
            'all_currency_amounts': '; '.join(page_data.get('financial_metrics', {}).get('currency_amounts', [])),
            'all_percentages': '; '.join(page_data.get('financial_metrics', {}).get('percentages', [])),
            'all_numbers': '; '.join(page_data.get('financial_metrics', {}).get('all_numbers', [])),
            'yoy_changes': '; '.join(page_data.get('financial_metrics', {}).get('yoy_changes', [])),
            
            # CHART INDICATORS
            'chart_indicators': '; '.join(page_data.get('chart_indicators', [])),
            'has_charts': len(page_data.get('chart_indicators', [])) > 0,
            
            # NUMERICAL RELATIONSHIPS
            'numerical_relationships_count': len(page_data.get('numerical_analysis', [])),
            
            # METADATA
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Add detailed financial metrics as separate fields
        financial_metrics = page_data.get('financial_metrics', {})
        
        # Capital metrics
        capital_metrics = financial_metrics.get('capital_metrics', [])
        for i, metric in enumerate(capital_metrics[:5]):  # Top 5
            main_row[f'capital_metric_{i+1}_name'] = metric.get('metric', '')
            main_row[f'capital_metric_{i+1}_value'] = metric.get('value', '')
        
        # Balance sheet items
        balance_items = financial_metrics.get('balance_sheet_items', [])
        for i, item in enumerate(balance_items[:5]):  # Top 5
            main_row[f'balance_item_{i+1}_name'] = item.get('metric', '')
            main_row[f'balance_item_{i+1}_value'] = item.get('value', '')
        
        # Performance metrics
        performance_metrics = financial_metrics.get('performance_metrics', [])
        for i, metric in enumerate(performance_metrics[:5]):  # Top 5
            main_row[f'performance_metric_{i+1}_name'] = metric.get('metric', '')
            main_row[f'performance_metric_{i+1}_value'] = metric.get('value', '')
        
        # Add table data as structured fields
        tables = page_data.get('tables', [])
        for table_idx, table in enumerate(tables[:3]):  # Top 3 tables
            main_row[f'table_{table_idx+1}_type'] = table.get('table_type', '')
            main_row[f'table_{table_idx+1}_rows'] = table.get('row_count', 0)
            main_row[f'table_{table_idx+1}_cols'] = table.get('col_count', 0)
            main_row[f'table_{table_idx+1}_text'] = table.get('table_text', '')[:500] + "..." if len(table.get('table_text', '')) > 500 else table.get('table_text', '')
            
            # Add numerical summary
            num_summary = table.get('numerical_summary', {})
            if num_summary:
                main_row[f'table_{table_idx+1}_number_count'] = num_summary.get('count', 0)
                main_row[f'table_{table_idx+1}_min_value'] = num_summary.get('min', '')
                main_row[f'table_{table_idx+1}_max_value'] = num_summary.get('max', '')
        
        rows.append(main_row)
    
    return pd.DataFrame(rows)

def install_pdfplumber():
    """Install pdfplumber"""
    try:
        st.info("Installing pdfplumber for comprehensive extraction...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        st.success("Successfully installed pdfplumber")
        return True
    except Exception as e:
        st.error(f"Failed to install pdfplumber: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="BOE ETL - Comprehensive Financial Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 BOE ETL - Comprehensive Financial Document Analysis")
    st.markdown("**Complete text extraction + Enhanced table/chart interpretation**")
    
    # Initialize parser
    parser = ComprehensiveFinancialParser()
    
    # Check capabilities
    if not parser.pdf_methods:
        st.warning("⚠️ Comprehensive parsing requires pdfplumber or PyMuPDF!")
        if st.button("🔧 Install pdfplumber"):
            if install_pdfplumber():
                st.experimental_rerun()
        return
    else:
        st.success(f"✅ Comprehensive parsing ready with: {', '.join(parser.pdf_methods)}")
    
    # File upload
    st.header("📁 Upload Financial Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload financial presentations, earnings reports, or regulatory filings"
    )
    
    if uploaded_file is not None:
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("🔄 Performing comprehensive extraction (ALL text + enhanced analysis)..."):
                    pages_data = parser.extract_comprehensive_data(tmp_path)
                
                st.success(f"✅ Comprehensive extraction completed: {uploaded_file.name} ({len(pages_data)} pages)")
                
                # Display results
                st.header("📊 Comprehensive Analysis Results")
                
                # Summary metrics
                total_words = sum(page['word_count'] for page in pages_data)
                total_tables = sum(len(page.get('tables', [])) for page in pages_data)
                total_chart_indicators = sum(len(page.get('chart_indicators', [])) for page in pages_data)
                total_financial_metrics = sum(
                    len(page.get('financial_metrics', {}).get('capital_metrics', [])) +
                    len(page.get('financial_metrics', {}).get('balance_sheet_items', [])) +
                    len(page.get('financial_metrics', {}).get('performance_metrics', []))
                    for page in pages_data
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages", len(pages_data))
                with col2:
                    st.metric("Total Words", total_words)
                with col3:
                    st.metric("Tables", total_tables)
                with col4:
                    st.metric("Financial Metrics", total_financial_metrics)
                
                # Page preview
                st.subheader("📄 Page Content Preview")
                
                if pages_data:
                    page_options = [f"Page {page['page']}" for page in pages_data]
                    selected_page = st.selectbox("Select page to preview:", page_options)
                    page_num = int(selected_page.split()[1])
                    
                    page_data = next(page for page in pages_data if page['page'] == page_num)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Page {page_num} - Complete Text Content:**")
                        st.text_area("Full Page Text:", page_data['full_text'], height=400)
                    
                    with col2:
                        st.write(f"**Page {page_num} - Enhanced Analysis:**")
                        
                        # Tables
                        tables = page_data.get('tables', [])
                        if tables:
                            st.write("**Tables Found:**")
                            for table in tables:
                                st.write(f"- {table['table_type']}: {table['row_count']}×{table['col_count']}")
                        
                        # Financial metrics
                        financial_metrics = page_data.get('financial_metrics', {})
                        st.write("**Financial Data:**")
                        st.write(f"- Currency amounts: {len(financial_metrics.get('currency_amounts', []))}")
                        st.write(f"- Percentages: {len(financial_metrics.get('percentages', []))}")
                        st.write(f"- YoY changes: {len(financial_metrics.get('yoy_changes', []))}")
                        
                        # Chart indicators
                        chart_indicators = page_data.get('chart_indicators', [])
                        if chart_indicators:
                            st.write("**Chart/Visual Elements:**")
                            for indicator in chart_indicators[:5]:
                                st.write(f"- {indicator}")
                
                # Generate comprehensive CSV
                st.header("📋 Comprehensive CSV Export")
                
                with st.spinner("🔄 Generating comprehensive CSV with ALL content..."):
                    df = create_comprehensive_csv(pages_data)
                
                st.success(f"✅ Generated comprehensive CSV with {len(df)} rows and {len(df.columns)} columns")
                st.info("📄 Each row contains the COMPLETE text of one page plus all enhanced analysis")
                
                # Display CSV preview
                st.subheader("CSV Preview - Complete Content + Enhanced Analysis")
                
                # Show key columns first
                key_columns = ['page_number', 'word_count', 'table_count', 'has_charts', 'capital_metrics_count']
                available_key_columns = [col for col in key_columns if col in df.columns]
                
                if available_key_columns:
                    st.write("**Key Metrics Preview:**")
                    st.dataframe(df[available_key_columns], use_container_width=True)
                
                st.write("**Full Dataset Preview (first 5 rows):**")
                st.dataframe(df.head(5), use_container_width=True)
                
                # Download button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comprehensive CSV (ALL Text + Enhanced Analysis)",
                    data=csv_data,
                    file_name=f"comprehensive_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Error in comprehensive processing: {str(e)}")
    
    # Information section
    st.header("ℹ️ About Comprehensive Financial Analysis")
    
    with st.expander("Complete Feature Set"):
        st.markdown("""
        ### 📄 Complete Text Extraction
        - **ALL Page Content**: Every word from every page preserved in full_page_text field
        - **Word/Character/Line Counts**: Complete text statistics
        - **No Content Loss**: 100% text preservation with enhanced analysis overlay
        
        ### 📊 Enhanced Structural Analysis
        - **Table Structure**: Headers, data types, numerical summaries
        - **Financial Metrics**: Capital ratios, balance sheet items, performance metrics
        - **Chart Detection**: Visual element indicators and chart components
        - **Numerical Relationships**: All numbers with context and relationships
        
        ### 🏦 Financial Intelligence
        - **Capital Metrics**: CET1, Tier 1, leverage ratios with values
        - **Balance Sheet**: Assets, liabilities, equity with amounts
        - **Performance**: ROA, ROE, revenue, net income
        - **YoY Analysis**: Year-over-year changes and trends
        
        ### 📈 Comprehensive Output
        - **Single Row Per Page**: Complete page content + all analysis in one row
        - **60+ Data Fields**: Every aspect of financial document analysis
        - **Structured + Unstructured**: Both complete text and parsed data
        - **Context Preservation**: Numbers linked to their meanings and relationships
        """)

if __name__ == "__main__":
    main()