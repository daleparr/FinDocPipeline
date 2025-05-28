#!/usr/bin/env python3
"""
Processing Module
=================

This module contains advanced text processing and feature extraction components
for the BoE ETL NLP extension package.

Components:
- feature_extraction: Main NLP processor for adding comprehensive features
- text_cleaning: Advanced text cleaning and normalization
- sentiment: Sentiment analysis for financial text
"""

from .feature_extraction import NLPProcessor, add_nlp_features

__all__ = [
    "NLPProcessor",
    "add_nlp_features",
]