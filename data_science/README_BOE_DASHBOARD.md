# Bank of England Supervisor Risk Assessment Dashboard

## 🏛️ Overview

The Bank of England Supervisor Risk Assessment Dashboard is a regulatory-grade financial institution risk analysis tool designed specifically for BoE supervisors. It provides comprehensive risk assessment with complete source attribution and traceability for regulatory compliance.

## 🚀 Quick Start

### Running the Dashboard

```bash
# Navigate to the data_science directory
cd data_science

# Install required dependencies
pip install streamlit pandas numpy plotly

# Launch the dashboard
streamlit run boe_supervisor_dashboard.py --server.port 8505
```

The dashboard will be available at: **http://localhost:8505**

## 📋 Key Features

### ✅ Complete Source Attribution & Traceability
- **Quarter Identification**: Every finding linked to specific quarters (e.g., "Q3 2024")
- **Speaker Attribution**: Exact roles identified (e.g., "Chief Risk Officer")
- **Document References**: Full document names and page numbers
- **Exact Quotes**: Verbatim statements with quotation marks
- **Context Information**: Situational context for each statement

### 🔍 Comprehensive Risk Analysis
- **Risk Score Attribution**: Detailed breakdown of score calculations
- **Topic-Driven Analysis**: Risk across Regulatory, Financial, Credit, Operational, Market categories
- **Sentiment Evolution**: Multi-quarter trend analysis with alerts
- **Contradiction Detection**: Automated detection of inconsistencies between presentation tone and financial data

### 🏛️ BoE-Specific Features
- **Professional Interface**: BoE blue color scheme and regulatory styling
- **Regulatory References**: Specific PRA/CRR article citations
- **Peer Comparison**: Benchmarking against major UK banks
- **Supervisory Actions**: Specific recommended next steps

### 📊 Advanced Analytics
- **Risk Score Breakdown**: Visual pie charts showing component contributions
- **Sentiment Trends**: Multi-line charts tracking sentiment evolution
- **Topic Analysis**: Comprehensive charts showing risk levels and trends
- **Peer Ranking**: Bar charts comparing institution against industry peers

## 📁 Document Upload

The dashboard supports multiple file formats:
- **PDF**: Earnings call transcripts, annual reports
- **TXT**: Plain text transcripts
- **DOCX**: Word documents
- **XLSX**: Excel spreadsheets with financial data

### Supported Institutions
- Barclays
- HSBC
- Lloyds Banking Group
- NatWest Group
- Standard Chartered
- Santander UK
- TSB Bank
- Metro Bank
- Virgin Money
- Nationwide Building Society
- Other (custom entry)

## 🎯 Analysis Configuration

### Available Analysis Options
- ✅ **Risk Score Attribution**: Detailed risk calculation breakdown
- ✅ **Contradiction Detection**: Identify inconsistencies between tone and data
- ✅ **Peer Comparison**: Compare against industry benchmarks
- ✅ **Regulatory Flags**: Identify potential regulatory concerns
- ✅ **Sentiment Analysis**: Track sentiment changes over time
- ✅ **Topic Analysis**: Comprehensive topic-driven risk assessment

## 📊 Dashboard Sections

### 1. Risk Score Attribution & Methodology
- Composite risk score with full calculation transparency
- Component breakdown (Topic Risk, Sentiment Risk, Trend Risk, Volatility Risk)
- Visual pie chart showing risk attribution
- Methodology explanation with audit trail

### 2. Topic-Driven Risk Analysis
- Risk levels across 5 major categories
- Sentiment analysis by topic
- Mention frequency tracking
- Risk trend analysis
- Detailed topic cards with:
  - Mentions across quarters
  - Average sentiment scores
  - Key indicators and risk drivers

### 3. Sentiment Evolution & Trend Analysis
- Multi-quarter sentiment tracking
- Trend lines for different risk categories
- Key sentiment insights with evidence
- Positive/negative zone indicators

### 4. Contradiction Detection
- Automated detection of inconsistencies
- Complete source attribution:
  - Quarter and speaker identification
  - Exact quotes from presentations
  - Supporting financial data
  - Discrepancy explanations
  - Supervisor notes and recommendations

### 5. Peer Comparison Analysis
- Risk score comparison against major UK banks
- Current ranking and percentile
- Industry average benchmarking
- Deviation from average analysis

### 6. Regulatory Attention Flags
- High/Medium/Low priority flags
- Specific regulatory references (PRA, CRR articles)
- Evidence and recommended actions
- Compliance tracking

### 7. Audit Trail & Evidence Documentation
- Complete processing history with timestamps
- Methodology transparency
- Export capabilities for regulatory compliance

## 📤 Export Options

### Available Exports
1. **Supervisor Report (JSON)**: Complete analysis results with metadata
2. **Analysis Data (JSON)**: Raw analysis data for further processing
3. **Audit Trail (JSON)**: Complete processing history for compliance

### Export File Naming Convention
```
supervisor_report_{institution}_{YYYYMMDD_HHMMSS}.json
audit_trail_{institution}_{YYYYMMDD_HHMMSS}.json
```

## 🔬 Methodology Transparency

### Risk Score Calculation
- **Topic Risk Weighting**: Regulatory (40%), Financial Performance (30%), Operations (20%), Market (10%)
- **Sentiment Integration**: Negative sentiment amplifies topic risk by 1.2x factor
- **Temporal Weighting**: Recent quarters weighted 2x vs historical data
- **Anomaly Detection**: Statistical outliers flagged using 2-sigma threshold

### Data Processing Pipeline
1. **Document Parsing**: Multi-format extraction with OCR validation
2. **NLP Processing**: Topic modeling using LDA with financial domain adaptation
3. **Entity Recognition**: Financial terms, figures, and speaker identification
4. **Sentiment Analysis**: VADER sentiment with financial context weighting

### Quality Assurance
- **Cross-Validation**: Results validated against known regulatory actions
- **Peer Comparison**: Scores normalized against industry benchmarks
- **Human Review**: Flagged cases require supervisor validation
- **Source Attribution**: Complete traceability to speaker, quarter, and document

## 🛡️ Regulatory Compliance

### Audit Trail Features
- Complete processing history with timestamps
- Methodology version tracking
- Source document attribution
- Analysis configuration logging
- Export history tracking

### Evidence Documentation
- Exact quotes with speaker attribution
- Quarter and document references
- Financial data sources
- Regulatory reference citations
- Supervisor notes and recommendations

## 🎭 Example Contradiction Analysis

```
🚨 Contradiction #1 - High Severity
Topic: Credit Quality
Quarter: Q3 2024
Speaker: Chief Risk Officer
Document: Q3_2024_Earnings_Call_Transcript.pdf
Timestamp: 14:23 into call

Presentation Tone: Optimistic
Exact Quote: "Credit quality remains strong with well-managed risk across all portfolios"

Financial Data: NPL ratio increased 40% QoQ, provisions up 25%
Source: Q3 2024 Financial Supplement, Page 12

Discrepancy: Presentation tone does not reflect deteriorating credit metrics
Supervisor Note: Requires immediate clarification on credit risk management
Regulatory Reference: IFRS 9, PRA SS1/18
```

## 🔧 Technical Requirements

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
```

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 📞 Support

For technical issues or questions about the dashboard:
1. Check the methodology section for calculation explanations
2. Review the audit trail for processing details
3. Verify document format compatibility
4. Ensure all required dependencies are installed

## 🔄 Version History

- **v2.1.0**: Enhanced BoE Supervisor Dashboard with complete source attribution
- **v2.0.0**: Added contradiction detection and peer comparison
- **v1.0.0**: Initial release with basic risk assessment

---

**Note**: This dashboard is designed for Bank of England supervisory use and provides regulatory-grade transparency and source attribution for confident supervisory decision-making.