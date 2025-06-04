import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import os
import re
import subprocess
import sys
import json

# Import the comprehensive parser from previous implementation
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
                    'financial_metrics': {},
                    'chart_indicators': []
                }
                
                # Extract tables with full structure
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_text = self._table_to_text(table)
                                page_data['tables'].append({
                                    'table_id': table_idx,
                                    'table_text': table_text,
                                    'row_count': len(table),
                                    'col_count': len(table[0]) if table else 0
                                })
                except Exception as e:
                    st.warning(f"Table extraction error on page {page_num + 1}: {str(e)}")
                
                # Enhanced financial analysis
                page_data['financial_metrics'] = self._extract_financial_data(full_text)
                page_data['chart_indicators'] = self._detect_chart_elements(full_text)
                
                pages_data.append(page_data)
        
        return pages_data
    
    def _extract_with_pymupdf(self, pdf_path):
        """Comprehensive extraction using PyMuPDF"""
        import fitz
        pages_data = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text = page.get_text()
            
            page_data = {
                'page': page_num + 1,
                'method': 'pymupdf_comprehensive',
                'full_text': full_text,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'line_count': len(full_text.split('\n')),
                'tables': [],
                'financial_metrics': {},
                'chart_indicators': []
            }
            
            page_data['financial_metrics'] = self._extract_financial_data(full_text)
            page_data['chart_indicators'] = self._detect_chart_elements(full_text)
            
            pages_data.append(page_data)
        
        doc.close()
        return pages_data
    
    def _table_to_text(self, table):
        """Convert table to readable text format"""
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            text_lines.append("\t".join(cleaned_row))
        
        return "\n".join(text_lines)
    
    def _extract_financial_data(self, text):
        """Extract financial metrics"""
        return {
            'currency_amounts': re.findall(r'\$\d+(?:,\d{3})*(?:\.\d+)?', text),
            'percentages': re.findall(r'\d+\.?\d*%', text),
            'all_numbers': re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        }
    
    def _detect_chart_elements(self, text):
        """Detect chart and visual elements"""
        indicators = []
        chart_terms = ['YoY', 'Walk', 'bps', 'basis points', 'Chart', 'Graph', 'Overview']
        
        for term in chart_terms:
            if term.lower() in text.lower():
                indicators.append(term)
        
        return indicators

class NLPDataProcessor:
    """Advanced NLP-ready data processor with theme classification"""
    
    def __init__(self):
        self.theme_patterns = self._initialize_theme_patterns()
    
    def _initialize_theme_patterns(self):
        """Initialize comprehensive financial theme patterns"""
        return {
            "Capital Adequacy": re.compile(
                r"(?:Capital\s+Adequacy|Capital\s+Strength|Capital\s+Position|Capital\s+Buffer)[^\n]*",
                re.IGNORECASE
            ),
            "Tier 1 Capital": re.compile(
                r"(?:Tier\s?1\s+Capital|Tier 1\s+Ratio|Core\s+Equity\s+Tier\s?1)[^\n]*",
                re.IGNORECASE
            ),
            "CET1 Capital": re.compile(
                r"(?:CET1|Core\s?Tier\s?1)\s+Capital[^\n]*",
                re.IGNORECASE
            ),
        }
    
    def create_raw_csv(self, pages_data):
        """Create raw comprehensive CSV"""
        rows = []
        
        for page_data in pages_data:
            page_num = page_data['page']
            
            # Combine all text content
            combined_text_parts = [page_data['full_text']]
            
            # Add table text
            for table in page_data.get('tables', []):
                if table.get('table_text'):
                    combined_text_parts.append(table['table_text'])
            
            combined_text = " ".join(combined_text_parts)
            
            row = {
                'document_id': f"page_{page_num}",
                'page_number': page_num,
                'extraction_method': page_data['method'],
                'full_page_text': page_data['full_text'],
                'table_count': len(page_data.get('tables', [])),
                'word_count': page_data['word_count'],
                'char_count': page_data['char_count'],
                'combined_text': combined_text,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            # Add table texts as separate columns
            for i, table in enumerate(page_data.get('tables', [])[:3]):  # Max 3 tables
                row[f'table_{i+1}_text'] = table.get('table_text', '')
            
            # Add financial metrics
            financial_metrics = page_data.get('financial_metrics', {})
            row['currency_amounts'] = '; '.join(financial_metrics.get('currency_amounts', []))
            row['percentages'] = '; '.join(financial_metrics.get('percentages', []))
            row['all_numbers'] = '; '.join(financial_metrics.get('all_numbers', []))
            
            # Add chart indicators
            row['chart_indicators'] = '; '.join(page_data.get('chart_indicators', []))
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def clean_for_nlp(self, df):
        """Clean and prepare data for NLP pipeline"""
        try:
            # 1. Identify text columns
            text_columns = [col for col in df.columns if col.lower().endswith('_text')]
            st.info(f"Identified text columns: {text_columns}")
            
            # 2. Combine all text columns
            df["combined_text"] = (
                df[text_columns]
                .fillna("")
                .agg(" ".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            
            # 3. Drop rows with no actual text
            original_count = len(df)
            df = df[df["combined_text"].str.len() > 0].copy()
            st.info(f"Removed {original_count - len(df)} empty rows")
            
            # 4. Create document ID
            if "document_id" in df.columns:
                df["doc_id"] = df["document_id"]
            else:
                df["doc_id"] = df.index.astype(str)
            
            # 5. Keep original text for metrics extraction
            df["clean_text"] = df["combined_text"]
            
            # 6. Select final columns for NLP
            nlp_columns = ["doc_id", "page_number", "clean_text", "word_count"]
            
            # Only include columns that exist
            available_columns = [col for col in nlp_columns if col in df.columns]
            output_df = df[available_columns].copy()
            
            return output_df, df  # Return both cleaned and full dataframes
            
        except Exception as e:
            st.error(f"Error in NLP cleaning: {str(e)}")
            return df, df

class EnhancedMetricsExtractor:
    """Enhanced metrics extractor with more flexible patterns and debugging"""
    
    def __init__(self):
        self.metric_patterns = self._initialize_enhanced_metric_patterns()
    
    def _initialize_enhanced_metric_patterns(self):
        """Initialize enhanced metric extraction patterns with more flexibility"""
        return {
            # Capital ratios - much more flexible patterns
            "CET1 Capital Ratio": [
                re.compile(r"CET1.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"(\d+\.?\d*)%.*?CET1", re.IGNORECASE | re.DOTALL),
                re.compile(r"Core Equity Tier 1.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
            ],
            "Tier 1 Capital Ratio": [
                re.compile(r"Tier\s*1.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"(\d+\.?\d*)%.*?Tier\s*1", re.IGNORECASE | re.DOTALL),
            ],
            "Total Capital Ratio": [
                re.compile(r"Total Capital.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"(\d+\.?\d*)%.*?Total Capital", re.IGNORECASE | re.DOTALL),
            ],
            "Leverage Ratio": [
                re.compile(r"Leverage.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"(\d+\.?\d*)%.*?Leverage", re.IGNORECASE | re.DOTALL),
            ],
            
            # Liquidity ratios
            "Liquidity Coverage Ratio": [
                re.compile(r"LCR.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"Liquidity Coverage.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
            ],
            
            # Profitability ratios
            "Return on Equity": [
                re.compile(r"ROE.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"Return on Equity.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
            ],
            "Return on Assets": [
                re.compile(r"ROA.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
                re.compile(r"Return on Assets.*?(\d+\.?\d*)%", re.IGNORECASE | re.DOTALL),
            ],
            
            # Asset amounts - very flexible
            "Total Assets": [
                re.compile(r"Total Assets.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
                re.compile(r"\$?(\d{1,3}(?:,\d{3})*).*?Total Assets", re.IGNORECASE | re.DOTALL),
            ],
            "Risk Weighted Assets": [
                re.compile(r"RWA.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
                re.compile(r"Risk.*?Weighted.*?Assets.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
            ],
            
            # Revenue and income
            "Net Revenue": [
                re.compile(r"Net Revenue.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
                re.compile(r"Revenue.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
            ],
            "Net Income": [
                re.compile(r"Net Income.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
                re.compile(r"Net Earnings.*?\$?(\d{1,3}(?:,\d{3})*)", re.IGNORECASE | re.DOTALL),
            ],
            
            # Simple number extraction near keywords
            "Book Value": [
                re.compile(r"Book Value.*?\$?(\d+\.?\d*)", re.IGNORECASE | re.DOTALL),
                re.compile(r"\$?(\d+\.?\d*).*?Book Value", re.IGNORECASE | re.DOTALL),
            ],
        }
    
    def extract_metrics_enhanced(self, nlp_df):
        """Enhanced metrics extraction with debugging"""
        rows = []
        debug_info = []
        
        for idx, row in nlp_df.iterrows():
            doc_id = row.get("doc_id", None)
            page_number = row.get("page_number", None)
            text_blob = row["clean_text"]
            
            page_matches = 0
            page_debug = {
                'page': page_number,
                'text_length': len(text_blob),
                'sample_text': text_blob[:300] + "..." if len(text_blob) > 300 else text_blob,
                'found_metrics': []
            }
            
            # For each metric type, try all patterns
            for metric_name, patterns in self.metric_patterns.items():
                for pattern_idx, pattern in enumerate(patterns):
                    try:
                        matches = pattern.findall(text_blob)
                        if matches:
                            for match in matches:
                                # Clean the matched value
                                clean_value = str(match).strip().replace(',', '').replace('$', '')
                                
                                # Validate it's a reasonable number
                                try:
                                    float_val = float(clean_value)
                                    if 0 <= float_val <= 1000000:  # Reasonable range
                                        rows.append({
                                            "doc_id": doc_id,
                                            "page_number": page_number,
                                            "metric_name": metric_name,
                                            "metric_value": clean_value,
                                            "pattern_used": pattern_idx,
                                            "extraction_timestamp": datetime.now().isoformat()
                                        })
                                        page_matches += 1
                                        page_debug['found_metrics'].append(f"{metric_name}: {clean_value}")
                                except ValueError:
                                    continue
                    except Exception as e:
                        continue
            
            page_debug['matches_found'] = page_matches
            debug_info.append(page_debug)
        
        # Build DataFrame in long form
        metrics_df = pd.DataFrame(rows)
        debug_df = pd.DataFrame(debug_info)
        
        return metrics_df, debug_df
    
    def create_wide_metrics(self, metrics_df):
        """Create wide-form metrics DataFrame"""
        if len(metrics_df) == 0:
            return pd.DataFrame()
        
        # Pivot to wide form
        wide_df = metrics_df.pivot_table(
            index=["doc_id", "page_number"],
            columns="metric_name",
            values="metric_value",
            aggfunc="first"
        ).reset_index()
        
        # Flatten column names
        wide_df.columns.name = None
        wide_df.columns = [str(col) for col in wide_df.columns]
        
        return wide_df

def install_pdfplumber():
    """Install pdfplumber"""
    try:
        st.info("Installing pdfplumber...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        st.success("Successfully installed pdfplumber")
        return True
    except Exception as e:
        st.error(f"Failed to install pdfplumber: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="BOE ETL - Debug Metrics Extraction",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 BOE ETL - Debug Metrics Extraction")
    st.markdown("**Enhanced metrics extraction with debugging and flexible patterns**")
    
    # Initialize components
    parser = ComprehensiveFinancialParser()
    nlp_processor = NLPDataProcessor()
    metrics_extractor = EnhancedMetricsExtractor()
    
    # Check capabilities
    if not parser.pdf_methods:
        st.warning("⚠️ Processing requires pdfplumber or PyMuPDF!")
        if st.button("🔧 Install pdfplumber"):
            if install_pdfplumber():
                st.experimental_rerun()
        return
    else:
        st.success(f"✅ Processing available with: {', '.join(parser.pdf_methods)}")
    
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
                with st.spinner("🔄 Extracting comprehensive data..."):
                    pages_data = parser.extract_comprehensive_data(tmp_path)
                
                with st.spinner("🔄 Creating raw structured data..."):
                    raw_df = nlp_processor.create_raw_csv(pages_data)
                
                with st.spinner("🤖 Cleaning and preparing for NLP..."):
                    nlp_df, full_df = nlp_processor.clean_for_nlp(raw_df)
                
                with st.spinner("🔍 Enhanced metrics extraction with debugging..."):
                    metrics_long_df, debug_df = metrics_extractor.extract_metrics_enhanced(nlp_df)
                    metrics_wide_df = metrics_extractor.create_wide_metrics(metrics_long_df)
                
                st.success(f"✅ Enhanced processing finished: {uploaded_file.name} ({len(pages_data)} pages)")
                
                # Display results
                st.header("🔍 Enhanced Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages Processed", len(pages_data))
                with col2:
                    st.metric("NLP-Ready Rows", len(nlp_df))
                with col3:
                    st.metric("Extracted Metrics", len(metrics_long_df))
                with col4:
                    unique_metrics = metrics_long_df['metric_name'].nunique() if len(metrics_long_df) > 0 else 0
                    st.metric("Unique Metric Types", unique_metrics)
                
                # Debug information
                st.subheader("🔍 Debug Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Extraction Debug Summary:**")
                    if len(debug_df) > 0:
                        total_matches = debug_df['matches_found'].sum()
                        pages_with_matches = (debug_df['matches_found'] > 0).sum()
                        avg_text_length = debug_df['text_length'].mean()
                        
                        st.write(f"- Total matches found: {total_matches}")
                        st.write(f"- Pages with matches: {pages_with_matches}/{len(debug_df)}")
                        st.write(f"- Average text length: {avg_text_length:.0f} chars")
                        
                        # Show pages with no matches
                        no_matches = debug_df[debug_df['matches_found'] == 0]
                        if len(no_matches) > 0:
                            st.write(f"- Pages with no matches: {len(no_matches)}")
                
                with col2:
                    st.write("**Sample Successful Extractions:**")
                    if len(metrics_long_df) > 0:
                        sample_metrics = metrics_long_df.head(10)[['metric_name', 'metric_value', 'page_number']]
                        st.dataframe(sample_metrics, use_container_width=True)
                    else:
                        st.write("No metrics extracted")
                
                # Metrics analysis
                if len(metrics_long_df) > 0:
                    st.subheader("📈 Extracted Financial Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Metrics by Type:**")
                        metric_counts = metrics_long_df['metric_name'].value_counts()
                        st.bar_chart(metric_counts.head(15))
                    
                    with col2:
                        st.write("**Pattern Usage:**")
                        if 'pattern_used' in metrics_long_df.columns:
                            pattern_counts = metrics_long_df['pattern_used'].value_counts()
                            st.write("Pattern effectiveness:")
                            for pattern_idx, count in pattern_counts.items():
                                st.write(f"- Pattern {pattern_idx}: {count} matches")
                
                # Data preview tabs
                st.subheader("📄 Data Preview")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Metrics (Long Form)", "Metrics (Wide Form)", "Debug Info", "Raw Data"])
                
                with tab1:
                    st.write("**Extracted metrics in long form (one row per metric):**")
                    if len(metrics_long_df) > 0:
                        st.dataframe(metrics_long_df.head(20), use_container_width=True)
                    else:
                        st.info("No metrics extracted from this document")
                
                with tab2:
                    st.write("**Extracted metrics in wide form (one row per page):**")
                    if len(metrics_wide_df) > 0:
                        st.dataframe(metrics_wide_df.head(10), use_container_width=True)
                    else:
                        st.info("No metrics extracted from this document")
                
                with tab3:
                    st.write("**Debug information for each page:**")
                    if len(debug_df) > 0:
                        st.dataframe(debug_df, use_container_width=True)
                    else:
                        st.info("No debug information available")
                
                with tab4:
                    st.write("**Raw extracted data with all original content:**")
                    st.dataframe(raw_df.head(10), use_container_width=True)
                
                # Download options
                st.header("📥 Download Enhanced Dataset")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_filename = uploaded_file.name.replace('.pdf', '')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Metrics Data")
                    
                    # Metrics Long Form CSV
                    if len(metrics_long_df) > 0:
                        metrics_long_csv = metrics_long_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Metrics CSV (Long Form)",
                            data=metrics_long_csv,
                            file_name=f"metrics_long_{base_filename}_{timestamp}.csv",
                            mime="text/csv"
                        )
                        
                        # Metrics Wide Form CSV
                        if len(metrics_wide_df) > 0:
                            metrics_wide_csv = metrics_wide_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Metrics CSV (Wide Form)",
                                data=metrics_wide_csv,
                                file_name=f"metrics_wide_{base_filename}_{timestamp}.csv",
                                mime="text/csv"
                            )
                        
                        # Metrics JSON
                        metrics_json = metrics_long_df.to_json(orient="records", force_ascii=False, indent=2)
                        st.download_button(
                            label="📥 Download Metrics JSON",
                            data=metrics_json,
                            file_name=f"metrics_{base_filename}_{timestamp}.json",
                            mime="application/json"
                        )
                    else:
                        st.info("No structured metrics found in this document")
                
                with col2:
                    st.subheader("🔍 Debug Data")
                    
                    # Debug CSV
                    if len(debug_df) > 0:
                        debug_csv = debug_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Debug CSV",
                            data=debug_csv,
                            file_name=f"debug_{base_filename}_{timestamp}.csv",
                            mime="text/csv"
                        )
                    
                    # Raw CSV
                    raw_csv = raw_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Raw CSV",
                        data=raw_csv,
                        file_name=f"raw_{base_filename}_{timestamp}.csv",
                        mime="text/csv"
                    )
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Error in enhanced processing: {str(e)}")
    
    # Information section
    st.header("ℹ️ About Enhanced Metrics Extraction")
    
    with st.expander("Debug Features"):
        st.markdown("""
        ### 🔍 Enhanced Pattern Matching
        - **Flexible Regex**: Multiple patterns per metric with DOTALL flag
        - **Bidirectional Matching**: Finds "CET1 13.4%" and "13.4% CET1"
        - **Context-Aware**: Uses broader text windows for matching
        - **Error Handling**: Robust processing with detailed error reporting
        
        ### 📊 Debug Information
        - **Page-by-Page Analysis**: Shows extraction success per page
        - **Pattern Effectiveness**: Tracks which patterns work best
        - **Sample Text**: Shows actual text being processed
        - **Match Validation**: Ensures extracted values are reasonable
        
        ### 🎯 Improved Metrics
        - **15+ Financial Metrics**: Capital ratios, profitability, assets
        - **Multiple Patterns**: 2-3 patterns per metric for better coverage
        - **Value Validation**: Filters out unreasonable numbers
        - **Context Preservation**: Links metrics to source pages
        """)

if __name__ == "__main__":
    main()