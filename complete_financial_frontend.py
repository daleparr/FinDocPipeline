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
            "Total Capital": re.compile(
                r"(?:Total\s+Capital\s+Ratio|Total\s+Regulatory\s+Capital)[^\n]*",
                re.IGNORECASE
            ),
            "Asset Quality": re.compile(
                r"(?:Asset\s+Quality|Asset[-\s]?Quality\s+Review|Loan\s+Quality)[^\n]*",
                re.IGNORECASE
            ),
            "Liquidity": re.compile(
                r"(?:Liquidity\s+Position|Liquidity\s+Coverage|Liquid\s+Assets|LCR)[^\n]*",
                re.IGNORECASE
            ),
            "Profitability": re.compile(
                r"(?:Profitability|Profit[-\s]?Growth|Earnings\s+Performance)[^\n]*",
                re.IGNORECASE
            ),
            "Return on Equity": re.compile(
                r"(?:Return\s+on\s+Equity|RoE)[^\n]*",
                re.IGNORECASE
            ),
            "Credit Risk": re.compile(
                r"(?:Credit\s+Risk|Credit[-\s]?Quality|Credit\s+Metrics)[^\n]*",
                re.IGNORECASE
            ),
            "Market Risk": re.compile(
                r"(?:Market\s+Risk|VaR|Value[-\s]?at[-\s]?Risk)[^\n]*",
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
            
            # 4. Normalize numeric tokens with preservation
            def normalize_numbers_advanced(text):
                # Preserve some context while normalizing
                text = re.sub(r"\$\s?([\d,]+(?:\.\d+)?)", r"<USD:\1>", text)  # Preserve amount
                text = re.sub(r"([\d,]+(?:\.\d+)?)\s?%", r"<PCT:\1>", text)  # Preserve percentage
                return text
            
            df["clean_text"] = df["combined_text"].apply(normalize_numbers_advanced)
            
            # 5. Create document ID
            if "document_id" in df.columns:
                df["doc_id"] = df["document_id"]
            else:
                df["doc_id"] = df.index.astype(str)
            
            # 6. Theme classification with confidence scoring
            def find_matched_themes_with_confidence(text):
                matches = []
                for theme, pattern in self.theme_patterns.items():
                    theme_matches = pattern.findall(text)
                    if theme_matches:
                        confidence = min(len(theme_matches) / 10.0, 1.0)  # Max confidence of 1.0
                        matches.append(f"{theme}({confidence:.2f})")
                return matches
            
            df["matched_themes"] = df["combined_text"].apply(find_matched_themes_with_confidence)
            df["theme_count"] = df["matched_themes"].apply(len)
            
            # 7. Data quality metrics
            df["text_quality_score"] = (
                (df["combined_text"].str.len() > 100).astype(int) * 0.3 +  # Length check
                (df["theme_count"] > 0).astype(int) * 0.4 +  # Theme relevance
                (df["combined_text"].str.contains(r'\d', regex=True)).astype(int) * 0.3  # Contains numbers
            )
            
            # 8. Select final columns for NLP
            nlp_columns = ["doc_id", "page_number", "clean_text", "matched_themes", 
                          "theme_count", "text_quality_score", "word_count"]
            
            # Only include columns that exist
            available_columns = [col for col in nlp_columns if col in df.columns]
            output_df = df[available_columns].copy()
            
            return output_df, df  # Return both cleaned and full dataframes
            
        except Exception as e:
            st.error(f"Error in NLP cleaning: {str(e)}")
            return df, df

class MetricsExtractor:
    """Extract specific financial metrics using regex patterns"""
    
    def __init__(self):
        self.metric_patterns = self._initialize_metric_patterns()
    
    def _initialize_metric_patterns(self):
        """Initialize comprehensive metric extraction patterns"""
        return {
            "CET1 Capital Ratio": re.compile(r"CET1\s+Capital\s+Ratio[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Tier 1 Capital Ratio": re.compile(r"Tier\s?1\s+Capital\s+Ratio[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Total Capital Ratio": re.compile(r"Total\s+Capital\s+Ratio[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Leverage Ratio": re.compile(r"Leverage\s+Ratio[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            
            "NPL Ratio": re.compile(r"(?:Non[-\s]?Performing\s+Loan|NPL)\s+Ratio[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Loan Loss Provisions": re.compile(r"(?:Loan\s+Loss\s+Provisions|LLP|Impairment\s+Charges)[^\n]*?\$?([\d,]+)", re.IGNORECASE),
            "Coverage Ratio": re.compile(r"Coverage\s+Ratio[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            
            "Liquidity Coverage Ratio": re.compile(r"(?:Liquidity\s+Coverage\s+Ratio|LCR)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Net Stable Funding Ratio": re.compile(r"(?:Net\s+Stable\s+Funding\s+Ratio|NSFR)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Loan-to-Deposit Ratio": re.compile(r"(?:Loan[-\s]?to[-\s]?Deposit\s+Ratio|LDR)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            
            "Return on Equity": re.compile(r"(?:Return\s+on\s+Equity|RoE)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Return on Assets": re.compile(r"(?:Return\s+on\s+Assets|RoA)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Cost-to-Income Ratio": re.compile(r"(?:Cost[-\s]?to[-\s]?Income\s+Ratio|C\/I)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            
            "Value-at-Risk": re.compile(r"(?:Value[-\s]?at[-\s]?Risk|VaR)[^\n]*?\$([\d,]+)", re.IGNORECASE),
            "Trading VaR Coverage": re.compile(r"Trading\s+VaR\s+Coverage[^\n]*?([\d]+)(?:\s+days?)", re.IGNORECASE),
            
            "Risk-Weighted Assets": re.compile(r"(?:Risk[-\s]?Weighted\s+Assets|RWA)[^\n]*?([\d]{1,3}(?:,\d{3})+)", re.IGNORECASE),
            "Total Assets": re.compile(r"Total\s+Assets[^\n]*?([\d]{1,3}(?:,\d{3})+)", re.IGNORECASE),
            
            "Customer Deposit Growth": re.compile(r"Customer\s+Deposit\s+Growth[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "Wholesale Funding Dependence": re.compile(r"Wholesale\s+Funding\s+Dependence[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            
            "Supplementary Leverage Ratio": re.compile(r"(?:Supplementary\s+Leverage\s+Ratio|SLR)[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            "CET1 Buffer": re.compile(r"CET1\s+Buffer\s+to\s+Minimum\s+Requirement[^\n]*?([\d]+(?:\.\d+)?%)", re.IGNORECASE),
            
            "CoCo Outstanding Amount": re.compile(r"CoCo\s+Outstanding\s+Amount[^\n]*?\$([\d,]+)", re.IGNORECASE),
            "Debt Maturity Profile": re.compile(r"Debt\s+Maturity\s+Profile[^\n]*?\$([\d,]+)", re.IGNORECASE),
        }
    
    def extract_metrics(self, nlp_df):
        """Extract metrics from NLP-ready dataframe"""
        rows = []
        
        for idx, row in nlp_df.iterrows():
            doc_id = row.get("doc_id", None)
            page_number = row.get("page_number", None)
            text_blob = row["clean_text"]
            
            # For each metric pattern, see if it matches in this text blob
            for metric_name, pattern in self.metric_patterns.items():
                match = pattern.search(text_blob)
                if match:
                    raw_value = match.group(1).strip()
                    rows.append({
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "metric_name": metric_name,
                        "metric_value": raw_value,
                        "extraction_timestamp": datetime.now().isoformat()
                    })
        
        # Build DataFrame in long form
        metrics_df = pd.DataFrame(rows)
        
        return metrics_df
    
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
        page_title="BOE ETL - Complete Financial Analysis",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 BOE ETL - Complete Financial Document Analysis")
    st.markdown("**Complete extraction + NLP cleaning + Structured metrics extraction**")
    
    # Initialize components
    parser = ComprehensiveFinancialParser()
    nlp_processor = NLPDataProcessor()
    metrics_extractor = MetricsExtractor()
    
    # Check capabilities
    if not parser.pdf_methods:
        st.warning("⚠️ Complete processing requires pdfplumber or PyMuPDF!")
        if st.button("🔧 Install pdfplumber"):
            if install_pdfplumber():
                st.experimental_rerun()
        return
    else:
        st.success(f"✅ Complete processing available with: {', '.join(parser.pdf_methods)}")
    
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
                
                with st.spinner("📊 Extracting structured metrics..."):
                    metrics_long_df = metrics_extractor.extract_metrics(nlp_df)
                    metrics_wide_df = metrics_extractor.create_wide_metrics(metrics_long_df)
                
                st.success(f"✅ Complete processing finished: {uploaded_file.name} ({len(pages_data)} pages)")
                
                # Display results
                st.header("📊 Complete Analysis Results")
                
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
                
                # Metrics analysis
                if len(metrics_long_df) > 0:
                    st.subheader("📈 Extracted Financial Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Metrics by Type:**")
                        metric_counts = metrics_long_df['metric_name'].value_counts()
                        st.bar_chart(metric_counts.head(10))
                    
                    with col2:
                        st.write("**Sample Extracted Metrics:**")
                        sample_metrics = metrics_long_df.head(10)[['metric_name', 'metric_value', 'page_number']]
                        st.dataframe(sample_metrics, use_container_width=True)
                
                # Data preview tabs
                st.subheader("📄 Data Preview")
                
                tab1, tab2, tab3, tab4 = st.tabs(["NLP-Ready Data", "Metrics (Long Form)", "Metrics (Wide Form)", "Raw Data"])
                
                with tab1:
                    st.write("**Cleaned and normalized data ready for NLP processing:**")
                    st.dataframe(nlp_df.head(10), use_container_width=True)
                
                with tab2:
                    st.write("**Extracted metrics in long form (one row per metric):**")
                    if len(metrics_long_df) > 0:
                        st.dataframe(metrics_long_df.head(20), use_container_width=True)
                    else:
                        st.info("No metrics extracted from this document")
                
                with tab3:
                    st.write("**Extracted metrics in wide form (one row per page):**")
                    if len(metrics_wide_df) > 0:
                        st.dataframe(metrics_wide_df.head(10), use_container_width=True)
                    else:
                        st.info("No metrics extracted from this document")
                
                with tab4:
                    st.write("**Raw extracted data with all original content:**")
                    st.dataframe(raw_df.head(10), use_container_width=True)
                
                # Download options
                st.header("📥 Download Complete Dataset")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_filename = uploaded_file.name.replace('.pdf', '')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 NLP & Text Data")
                    
                    # NLP-Ready CSV
                    nlp_csv = nlp_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download NLP-Ready CSV",
                        data=nlp_csv,
                        file_name=f"nlp_ready_{base_filename}_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Raw CSV
                    raw_csv = raw_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Raw Complete CSV",
                        data=raw_csv,
                        file_name=f"raw_complete_{base_filename}_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.subheader("📊 Structured Metrics")
                    
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
            
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Error in complete processing: {str(e)}")
    
    # Information section
    st.header("ℹ️ About Complete Financial Analysis")
    
    with st.expander("Complete Feature Set"):
        st.markdown("""
        ### 📄 Complete Text Extraction
        - **ALL Page Content**: Every word from every page preserved
        - **Enhanced Table/Chart Analysis**: Structured interpretation of visual elements
        - **Page-by-Page Processing**: Detailed analysis of each page
        
        ### 🤖 NLP-Ready Processing
        - **Text Consolidation**: Combines all text sources into unified content
        - **Smart Normalization**: `<USD:2,412>` and `<PCT:13.4>` tokens with value preservation
        - **Theme Classification**: Financial themes with confidence scoring
        - **Quality Scoring**: Automated assessment for NLP readiness
        
        ### 📊 Structured Metrics Extraction
        - **20+ Financial Metrics**: CET1, Tier 1, NPL ratios, ROE, VaR, etc.
        - **Multiple Output Formats**: Long form, wide form, CSV, and JSON
        - **Precise Pattern Matching**: Regex patterns for accurate extraction
        - **Context Preservation**: Links metrics to their source pages
        
        ### 📥 Download Options
        - **NLP-Ready CSV**: Cleaned, normalized text for ML/AI processing
        - **Raw Complete CSV**: All extracted content with original formatting
        - **Metrics CSV (Long Form)**: One row per extracted metric
        - **Metrics CSV (Wide Form)**: One row per page with metrics as columns
        - **Metrics JSON**: Structured data for API integration
        """)

if __name__ == "__main__":
    main()