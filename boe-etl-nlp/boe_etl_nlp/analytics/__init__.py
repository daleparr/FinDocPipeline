#!/usr/bin/env python3
"""
Analytics Module
================

This module contains advanced analytics components for the BoE ETL NLP extension,
including topic modeling, sentiment analysis, and financial classification.

Components:
- topic_modeling: Hybrid topic modeling with seed themes and BERTopic
- sentiment: Financial sentiment analysis
- classification: Financial data classification
"""

from .topic_modeling import TopicModeler, get_topic_modeler
from .sentiment import SentimentAnalyzer, analyze_sentiment
from .classification import FinancialClassifier, classify_document_type

__all__ = [
    "TopicModeler",
    "get_topic_modeler",
    "SentimentAnalyzer",
    "analyze_sentiment",
    "FinancialClassifier",
    "classify_document_type",
]