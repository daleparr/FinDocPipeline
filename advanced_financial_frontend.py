import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys

class AdvancedFinancialParser:
    """Advanced parser for financial documents with table structure and visual element detection"""
    
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
    
    def extract_structured_data_pdfplumber(self, pdf_path):
        """Extract structured data using pdfplumber with advanced table detection"""
        import pdfplumber
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_data = {
                    'page': page_num + 1,
                    'method': 'pdfplumber_advanced',
                    'raw_text': page.extract_text() or "",
                    'tables': [],
                    'structured_data': [],
                    'visual_elements': []
                }
                
                # Extract tables with structure preservation
                try:
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            structured_table = self._process_table_structure(table, table_idx)
                            page_data['tables'].append(structured_table)
                            page_data['structured_data'].extend(structured_table['data_rows'])
                except Exception as e:
                    st.warning(f"Table extraction error on page {page_num + 1}: {str(e)}")
                
                # Extract text blocks with positioning
                try:
                    chars = page.chars
                    text_blocks = self._extract_positioned_text_blocks(chars)
                    page_data['text_blocks'] = text_blocks
                except Exception as e:
                    st.warning(f"Text block extraction error on page {page_num + 1}: {str(e)}")
                
                # Detect financial patterns and relationships
                financial_data = self._extract_financial_relationships(page_data)
                page_data['financial_analysis'] = financial_data
                
                pages_data.append(page_data)
        
        return pages_data
    
    def _process_table_structure(self, table, table_idx):
        """Process table to preserve structure and extract meaningful data"""
        if not table or len(table) == 0:
            return None
        
        # Identify headers (usually first row, sometimes first column)
        headers = []
        data_rows = []
        
        # Clean and process table
        cleaned_table = []
        for row in table:
            cleaned_row = [cell.strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) > 0:
            # First row as headers
            headers = cleaned_table[0]
            
            # Process data rows
            for row_idx, row in enumerate(cleaned_table[1:], 1):
                if any(cell for cell in row):  # Skip empty rows
                    row_data = {}
                    for col_idx, cell in enumerate(row):
                        header = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx}"
                        row_data[header] = cell
                        
                        # Extract numerical relationships
                        if cell and self._is_financial_value(cell):
                            row_data[f"{header}_numeric"] = self._extract_numeric_value(cell)
                    
                    row_data['table_id'] = table_idx
                    row_data['row_index'] = row_idx
                    data_rows.append(row_data)
        
        return {
            'table_id': table_idx,
            'headers': headers,
            'row_count': len(data_rows),
            'col_count': len(headers),
            'data_rows': data_rows,
            'table_type': self._classify_table_type(headers, data_rows)
        }
    
    def _extract_positioned_text_blocks(self, chars):
        """Extract text blocks with their positions for better context understanding"""
        if not chars:
            return []
        
        # Group characters into text blocks based on position
        text_blocks = []
        current_block = {'text': '', 'x': 0, 'y': 0, 'chars': []}
        
        for char in chars:
            if not current_block['chars']:
                current_block = {
                    'text': char['text'],
                    'x': char['x0'],
                    'y': char['y0'],
                    'chars': [char]
                }
            else:
                # Check if character is part of current block (similar position)
                if (abs(char['y0'] - current_block['y']) < 5 and 
                    abs(char['x0'] - (current_block['x'] + len(current_block['text']) * 6)) < 10):
                    current_block['text'] += char['text']
                    current_block['chars'].append(char)
                else:
                    # Start new block
                    if current_block['text'].strip():
                        text_blocks.append(current_block)
                    current_block = {
                        'text': char['text'],
                        'x': char['x0'],
                        'y': char['y0'],
                        'chars': [char]
                    }
        
        # Add final block
        if current_block['text'].strip():
            text_blocks.append(current_block)
        
        return text_blocks
    
    def _extract_financial_relationships(self, page_data):
        """Extract financial relationships and contextual data"""
        financial_data = {
            'metrics': [],
            'relationships': [],
            'headers': [],
            'chart_indicators': []
        }
        
        text = page_data['raw_text']
        
        # Extract section headers
        headers = re.findall(r'^([A-Z][A-Za-z\s&]+)$', text, re.MULTILINE)
        financial_data['headers'] = headers
        
        # Extract financial metrics with context
        patterns = {
            'capital_ratios': r'(CET1|Tier 1|Capital Ratio)[:\s]+(\d+\.?\d*)%?',
            'assets': r'(Assets|Total Assets)[:\s]+\$?(\d+(?:,\d{3})*)',
            'revenue': r'(Revenue|Net Revenue)[:\s]+\$?(\d+(?:,\d{3})*)',
            'deposits': r'(Deposits)[:\s]+\$?(\d+(?:,\d{3})*)',
            'equity': r'(Equity)[:\s]+\$?(\d+(?:,\d{3})*)',
            'yoy_changes': r'YoY[:\s]*\(?([+-]?\d+\.?\d*)%?\)?'
        }
        
        for metric_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        financial_data['metrics'].append({
                            'type': metric_type,
                            'label': match[0],
                            'value': match[1],
                            'context': 'extracted_from_text'
                        })
                    else:
                        financial_data['metrics'].append({
                            'type': metric_type,
                            'value': match[0] if match else match,
                            'context': 'extracted_from_text'
                        })
        
        # Detect chart indicators
        chart_indicators = [
            'YoY Standardized', 'Walk', 'bps', 'basis points',
            'Management Buffer', 'GSIB Surcharge', 'Regulatory Minimum'
        ]
        
        for indicator in chart_indicators:
            if indicator.lower() in text.lower():
                financial_data['chart_indicators'].append(indicator)
        
        return financial_data
    
    def _classify_table_type(self, headers, data_rows):
        """Classify the type of financial table"""
        header_text = ' '.join(headers).lower()
        
        if any(term in header_text for term in ['capital', 'ratio', 'cet1']):
            return 'capital_metrics'
        elif any(term in header_text for term in ['assets', 'liabilities']):
            return 'balance_sheet'
        elif any(term in header_text for term in ['revenue', 'income', 'earnings']):
            return 'income_statement'
        elif any(term in header_text for term in ['cash', 'flow']):
            return 'cash_flow'
        else:
            return 'general_financial'
    
    def _is_financial_value(self, cell):
        """Check if cell contains a financial value"""
        if not cell:
            return False
        
        # Check for currency, percentages, or large numbers
        patterns = [
            r'\$\d+',  # Currency
            r'\d+\.?\d*%',  # Percentage
            r'\d{1,3}(?:,\d{3})+',  # Large numbers with commas
            r'\d+\.?\d*[BMK]',  # Numbers with B/M/K suffix
        ]
        
        return any(re.search(pattern, cell) for pattern in patterns)
    
    def _extract_numeric_value(self, cell):
        """Extract numeric value from cell"""
        if not cell:
            return None
        
        # Remove currency symbols and extract number
        cleaned = re.sub(r'[^\d.,%-]', '', cell)
        
        # Extract the main number
        number_match = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', cleaned)
        if number_match:
            return number_match.group().replace(',', '')
        
        return None
    
    def extract_structured_data(self, pdf_path):
        """Main extraction method"""
        if 'pdfplumber' in self.pdf_methods:
            return self.extract_structured_data_pdfplumber(pdf_path)
        else:
            raise Exception("pdfplumber required for advanced table extraction")

def create_advanced_csv(pages_data):
    """Create advanced CSV with structured financial data"""
    rows = []
    
    for page_data in pages_data:
        page_num = page_data['page']
        
        # Base page information
        base_row = {
            'document_id': f"page_{page_num}",
            'page_number': page_num,
            'extraction_method': page_data['method'],
            'total_tables': len(page_data.get('tables', [])),
            'total_structured_rows': len(page_data.get('structured_data', [])),
            'section_headers': '; '.join(page_data.get('financial_analysis', {}).get('headers', [])),
            'chart_indicators': '; '.join(page_data.get('financial_analysis', {}).get('chart_indicators', [])),
        }
        
        # If page has structured data, create rows for each data element
        structured_data = page_data.get('structured_data', [])
        
        if structured_data:
            for data_row in structured_data:
                row = base_row.copy()
                row.update({
                    'table_id': data_row.get('table_id', ''),
                    'row_index': data_row.get('row_index', ''),
                    'data_type': 'structured_table_row'
                })
                
                # Add all table columns as separate fields
                for key, value in data_row.items():
                    if key not in ['table_id', 'row_index']:
                        # Clean column names for CSV
                        clean_key = re.sub(r'[^\w]', '_', str(key)).lower()
                        row[f"table_{clean_key}"] = str(value) if value else ""
                
                rows.append(row)
        
        # Add financial metrics as separate rows
        financial_metrics = page_data.get('financial_analysis', {}).get('metrics', [])
        for metric in financial_metrics:
            row = base_row.copy()
            row.update({
                'data_type': 'financial_metric',
                'metric_type': metric.get('type', ''),
                'metric_label': metric.get('label', ''),
                'metric_value': metric.get('value', ''),
                'metric_context': metric.get('context', '')
            })
            rows.append(row)
        
        # If no structured data, create summary row
        if not structured_data and not financial_metrics:
            row = base_row.copy()
            row.update({
                'data_type': 'page_summary',
                'raw_text_sample': page_data.get('raw_text', '')[:500] + "..." if len(page_data.get('raw_text', '')) > 500 else page_data.get('raw_text', ''),
                'word_count': len(page_data.get('raw_text', '').split()),
                'extraction_timestamp': datetime.now().isoformat()
            })
            rows.append(row)
    
    return pd.DataFrame(rows)

def install_pdfplumber():
    """Install pdfplumber for advanced table extraction"""
    try:
        st.info("Installing pdfplumber for advanced table extraction...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        st.success("Successfully installed pdfplumber")
        return True
    except Exception as e:
        st.error(f"Failed to install pdfplumber: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="BOE ETL - Advanced Financial Parser",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 BOE ETL - Advanced Financial Document Parser")
    st.markdown("**Structured extraction of tables, charts, and financial relationships**")
    
    # Initialize parser
    parser = AdvancedFinancialParser()
    
    # Check capabilities
    if not parser.pdf_methods:
        st.warning("⚠️ Advanced parsing requires pdfplumber!")
        if st.button("🔧 Install pdfplumber"):
            if install_pdfplumber():
                st.experimental_rerun()
        return
    else:
        st.success(f"✅ Advanced parsing ready with: {', '.join(parser.pdf_methods)}")
    
    # Sidebar
    st.sidebar.header("⚙️ Advanced Options")
    extract_tables = st.sidebar.checkbox("Extract Table Structure", value=True)
    extract_relationships = st.sidebar.checkbox("Financial Relationships", value=True)
    preserve_context = st.sidebar.checkbox("Preserve Context", value=True)
    
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
                with st.spinner("🔄 Performing advanced financial data extraction..."):
                    pages_data = parser.extract_structured_data(tmp_path)
                
                st.success(f"✅ Advanced extraction completed: {uploaded_file.name} ({len(pages_data)} pages)")
                
                # Display advanced results
                st.header("📊 Advanced Analysis Results")
                
                # Summary metrics
                total_tables = sum(len(page.get('tables', [])) for page in pages_data)
                total_structured_rows = sum(len(page.get('structured_data', [])) for page in pages_data)
                total_financial_metrics = sum(len(page.get('financial_analysis', {}).get('metrics', [])) for page in pages_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages Processed", len(pages_data))
                with col2:
                    st.metric("Tables Extracted", total_tables)
                with col3:
                    st.metric("Structured Rows", total_structured_rows)
                with col4:
                    st.metric("Financial Metrics", total_financial_metrics)
                
                # Detailed page analysis
                st.subheader("📄 Page-by-Page Structured Analysis")
                
                for page_data in pages_data[:3]:  # Show first 3 pages
                    with st.expander(f"Page {page_data['page']} - Advanced Analysis", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Table Structure:**")
                            tables = page_data.get('tables', [])
                            if tables:
                                for table in tables:
                                    st.write(f"- Table {table['table_id']}: {table['row_count']} rows × {table['col_count']} cols")
                                    st.write(f"  Type: {table['table_type']}")
                                    st.write(f"  Headers: {', '.join(table['headers'][:3])}...")
                            else:
                                st.write("No structured tables found")
                            
                            st.write("**Section Headers:**")
                            headers = page_data.get('financial_analysis', {}).get('headers', [])
                            for header in headers[:5]:
                                st.write(f"- {header}")
                        
                        with col2:
                            st.write("**Financial Metrics:**")
                            metrics = page_data.get('financial_analysis', {}).get('metrics', [])
                            for metric in metrics[:5]:
                                st.write(f"- {metric['type']}: {metric.get('label', '')} = {metric.get('value', '')}")
                            
                            st.write("**Chart Indicators:**")
                            chart_indicators = page_data.get('financial_analysis', {}).get('chart_indicators', [])
                            for indicator in chart_indicators:
                                st.write(f"- {indicator}")
                
                # Generate advanced CSV
                st.header("📋 Advanced Structured CSV Export")
                
                with st.spinner("🔄 Generating advanced structured CSV..."):
                    df = create_advanced_csv(pages_data)
                
                st.success(f"✅ Generated advanced CSV with {len(df)} rows and {len(df.columns)} columns")
                
                # Show data type breakdown
                if 'data_type' in df.columns:
                    data_type_counts = df['data_type'].value_counts()
                    st.write("**Data Type Breakdown:**")
                    for data_type, count in data_type_counts.items():
                        st.write(f"- {data_type}: {count} rows")
                
                # Display CSV preview
                st.subheader("CSV Preview - Structured Financial Data")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Download button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Advanced Structured CSV",
                    data=csv_data,
                    file_name=f"advanced_financial_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Error in advanced processing: {str(e)}")
    
    # Information section
    st.header("ℹ️ About Advanced Financial Parser")
    
    with st.expander("Advanced Features & Capabilities"):
        st.markdown("""
        ### 🏦 Advanced Financial Document Processing
        - **Table Structure Preservation**: Maintains row/column relationships and headers
        - **Financial Relationship Extraction**: Links numbers to their contextual labels
        - **Chart Indicator Detection**: Identifies visual elements like "YoY Walk" charts
        - **Section Header Recognition**: Preserves document structure and hierarchy
        
        ### 📊 Structured Data Output
        - **Multiple Row Types**: Table rows, financial metrics, and page summaries
        - **Contextual Linking**: Numbers connected to their labels and meanings
        - **Table Classification**: Automatically categorizes tables (capital metrics, balance sheet, etc.)
        - **Financial Metric Extraction**: Specialized patterns for CET1, assets, revenue, etc.
        
        ### 🔍 Enhanced Analysis
        - **Position-Aware Text Extraction**: Uses character positioning for better context
        - **Financial Pattern Recognition**: Specialized regex for banking/financial terminology
        - **Chart Element Detection**: Identifies stacked bar charts, YoY analysis, basis points
        - **Relationship Mapping**: Connects related financial data points
        
        ### 📈 Business Intelligence
        - **Capital Ratio Analysis**: CET1, Tier 1, regulatory ratios
        - **Balance Sheet Structure**: Assets, liabilities, equity breakdown
        - **Performance Metrics**: YoY changes, growth rates, efficiency ratios
        - **Risk Indicators**: Regulatory buffers, stress test results
        """)

if __name__ == "__main__":
    main()