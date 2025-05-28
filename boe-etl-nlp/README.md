# BoE ETL NLP Extension

Advanced NLP capabilities for the Bank of England ETL pipeline, providing topic modeling, sentiment analysis, and financial text classification.

## Overview

The `boe-etl-nlp` package extends the core `boe-etl` pipeline with sophisticated natural language processing features specifically designed for financial documents and earnings calls. It extracts comprehensive NLP features, performs topic modeling, and provides analytical dashboards.

## Features

### 🔍 **NLP Feature Extraction**
- **Financial Term Detection**: Comprehensive extraction of financial vocabulary
- **Figure Extraction**: Automatic detection of financial figures and amounts
- **Actual vs Projection Classification**: Linguistic analysis to distinguish historical data from projections
- **Speaker Analysis**: Management vs analyst identification
- **Temporal Indicators**: Time-based language detection

### 📊 **Topic Modeling**
- **Hybrid Approach**: Combines seed themes with BERTopic for emerging topic discovery
- **Financial Themes**: Pre-configured themes for banking and finance
- **Confidence Scoring**: Reliability metrics for topic assignments

### 💭 **Sentiment Analysis**
- **Financial Context**: Sentiment analysis tuned for financial language
- **Confidence Metrics**: Reliability scoring for sentiment classifications

### 🏷️ **Document Classification**
- **Document Type Detection**: Earnings calls, presentations, reports, etc.
- **Content Categorization**: Financial performance, risk factors, strategy, etc.
- **Statement Type Classification**: Income statement, balance sheet, cash flow, etc.

### 📈 **Visualization & Dashboards**
- **Interactive Charts**: Topic distribution, sentiment analysis
- **Summary Statistics**: Comprehensive analytics overview
- **Streamlit Integration**: Ready-to-use web dashboards

## Installation

```bash
# Install from source
cd boe-etl-nlp
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Dependencies

**Core Requirements:**
- `boe-etl>=1.0.0` (core ETL functionality)
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `scikit-learn>=1.3.0`

**Optional Advanced Features:**
- `bertopic>=0.15.0` (advanced topic modeling)
- `transformers>=4.21.0` (transformer models)
- `plotly>=5.15.0` (interactive visualizations)
- `streamlit>=1.25.0` (web dashboards)

## Quick Start

### Basic NLP Feature Extraction

```python
from boe_etl import ETLPipeline
from boe_etl_nlp import NLPProcessor

# Process document with core ETL
pipeline = ETLPipeline()
results = pipeline.process_document('earnings.pdf', 'JPMorgan', 'Q1_2025')
df = pipeline.to_dataframe(results)

# Add comprehensive NLP features
nlp_processor = NLPProcessor()
enhanced_df = nlp_processor.add_nlp_features(df)

print(f"Added NLP features: {list(enhanced_df.columns)}")
```

### Topic Modeling

```python
from boe_etl_nlp import TopicModeler

# Analyze topics in financial data
modeler = TopicModeler()
records = df.to_dict('records')
topic_results = modeler.process_batch(records, 'JPMorgan', 'Q1_2025')

# View topic assignments
for record in topic_results[:3]:
    print(f"Text: {record['text'][:50]}...")
    print(f"Topic: {record['topic_label']}")
    print(f"Confidence: {record['topic_confidence']:.2f}")
    print("---")
```

### Sentiment Analysis

```python
from boe_etl_nlp import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze_sentiment(
    "We reported strong revenue growth of 15% this quarter"
)

print(f"Sentiment: {sentiment['sentiment_label']}")
print(f"Score: {sentiment['sentiment_score']:.2f}")
print(f"Confidence: {sentiment['confidence']:.2f}")
```

### Document Classification

```python
from boe_etl_nlp import FinancialClassifier

classifier = FinancialClassifier()
doc_type = classifier.classify_document_type(
    text="Welcome to our Q1 earnings call...",
    filename="JPMorgan_Q1_2025_earnings_call.txt"
)

print(f"Document Type: {doc_type['document_type']}")
print(f"Confidence: {doc_type['confidence']:.2f}")
```

### Interactive Dashboard

```python
from boe_etl_nlp import create_dashboard
import streamlit as st

# Create and render dashboard
dashboard = create_dashboard(enhanced_df)
dashboard.render_streamlit_dashboard(enhanced_df)
```

## Command Line Interface

The package includes a CLI for batch processing:

```bash
# Add NLP features to a CSV file
boe-etl-nlp process data.csv --output enhanced_data.csv

# Analyze topics in financial data
boe-etl-nlp topics data.csv --bank "JPMorgan" --quarter "Q1_2025"

# Process with verbose logging
boe-etl-nlp process data.csv --verbose
```

## NLP Features Reference

The `add_nlp_features()` method adds 25+ NLP features to your data:

### Financial Analysis
- `all_financial_terms`: Extracted financial vocabulary
- `financial_figures`: Detected monetary amounts and percentages
- `has_financial_terms`: Boolean flag for financial content
- `has_financial_figures`: Boolean flag for numerical data

### Data Classification
- `data_type`: actual | projection | unclear | unknown
- `is_actual_data`: Boolean flag for historical data
- `is_projection_data`: Boolean flag for forward-looking statements

### Topic Analysis
- `primary_topic`: Assigned topic category
- `has_financial_topic`: Revenue, Capital, Risk topics
- `has_strategy_topic`: Strategic content flag
- `has_operational_topic`: Operational efficiency flag

### Speaker Analysis
- `is_management`: CEO, CFO, Chief officers
- `is_analyst`: Financial analysts
- `is_named_speaker`: Known vs unknown speakers

### Text Metrics
- `word_count`: Number of words
- `char_count`: Character count
- `temporal_indicators`: Time-based language
- `processing_date`: Processing timestamp

## Architecture

```
boe-etl-nlp/
├── boe_etl_nlp/
│   ├── processing/          # Feature extraction
│   │   └── feature_extraction.py
│   ├── analytics/           # Advanced analytics
│   │   ├── topic_modeling.py
│   │   ├── sentiment.py
│   │   └── classification.py
│   ├── visualization/       # Dashboards & charts
│   │   └── dashboard.py
│   └── cli.py              # Command line interface
├── setup.py
└── README.md
```

## Integration with Core ETL

The NLP extension is designed to work seamlessly with the core `boe-etl` package:

1. **Core ETL** handles document parsing, data extraction, and standardization
2. **NLP Extension** adds advanced text analysis and feature extraction
3. **Combined Output** provides both structured data and rich NLP insights

## Configuration

Customize behavior with configuration dictionaries:

```python
config = {
    'seed_themes_path': 'custom_themes.yml',
    'financial_terms_threshold': 3,
    'sentiment_model': 'finbert',
    'topic_confidence_threshold': 0.7
}

processor = NLPProcessor(config)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [boe-etl repository](https://github.com/daleparr/boe-etl)
- Documentation: See inline docstrings and examples
- Email: etl-team@bankofengland.co.uk