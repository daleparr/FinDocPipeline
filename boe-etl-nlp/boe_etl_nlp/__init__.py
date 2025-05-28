#!/usr/bin/env python3
"""
BoE ETL NLP Extension Package
=============================

This package provides NLP extensions for the boe-etl core package,
including topic modeling, sentiment analysis, and advanced text processing.

Main Components:
- analytics: Topic modeling, sentiment analysis, financial classification
- processing: Advanced text cleaning and feature extraction
- schema: NLP-enhanced data schemas
- visualization: Analytics dashboards and charts
- frontend: NLP-enhanced web interface

Example Usage:
    >>> from boe_etl import ETLPipeline
    >>> from boe_etl_nlp import NLPProcessor
    >>> 
    >>> # Extract data with core ETL
    >>> pipeline = ETLPipeline()
    >>> results = pipeline.process_document('earnings.pdf', 'JPMorgan', 'Q1_2025')
    >>> df = pipeline.to_dataframe(results)
    >>> 
    >>> # Add NLP features
    >>> nlp_processor = NLPProcessor()
    >>> enhanced_df = nlp_processor.add_nlp_features(df)
    >>> topics = nlp_processor.analyze_topics(enhanced_df)
"""

__version__ = "1.0.0"
__author__ = "Bank of England ETL Team"
__email__ = "etl-team@bankofengland.co.uk"

# Import main classes for easy access
from .processing.feature_extraction import NLPProcessor, add_nlp_features
from .analytics.topic_modeling import TopicModeler, get_topic_modeler
from .analytics.sentiment import SentimentAnalyzer, analyze_sentiment
from .analytics.classification import FinancialClassifier, classify_document_type
from .visualization.dashboard import NLPDashboard, create_dashboard

# Define what gets imported with "from boe_etl_nlp import *"
__all__ = [
    "NLPProcessor",
    "add_nlp_features",
    "TopicModeler",
    "get_topic_modeler",
    "SentimentAnalyzer",
    "analyze_sentiment",
    "FinancialClassifier",
    "classify_document_type",
    "NLPDashboard",
    "create_dashboard",
]

# Package metadata
PACKAGE_INFO = {
    "name": "boe-etl-nlp",
    "version": __version__,
    "description": "NLP extensions for the BoE ETL pipeline",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/daleparr/boe-etl",
    "license": "MIT",
}