# PRA Risk Analysis System

An AI-driven risk analysis system for the Prudential Regulation Authority (PRA) that analyzes unstructured financial data to provide proactive risk insights.

## Project Overview

This project develops an AI-driven system to analyze unstructured financial data (earnings-call transcripts, presentations) to identify early warning signals of financial risk, enabling proactive intervention by regulators.

## Key Components

### Data Processing
- Multi-modal ETL pipeline for earnings-call transcripts and PDF presentations
- Data cleaning and preprocessing
- Parquet dataset storage

### Analysis
- Dynamic topic modeling using BERTopic and FinBERT
- Sentiment analysis with drift detection
- Contradiction analysis between narratives and financial metrics
- Peer comparison analysis

### Visualization & Alerts
- Interactive dashboard for risk visualization
- Real-time alert system via Slack/email
- Explainable AI components with SHAP/LIME

## Project Structure

```
src/
├── data/           # Raw and processed data
├── notebooks/      # Jupyter notebooks for analysis
├── etl/           # Data ingestion and processing scripts
├── models/        # ML/NLP model implementations
├── visualization/ # Dashboard and visualization code
├── alerts/        # Alert generation and notification system
└── utils/         # Utility functions and helpers

docs/             # Project documentation
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- Git
- Jupyter Notebook/Lab
- Required Python packages (see requirements.txt)

## License

[Add appropriate license information here]
