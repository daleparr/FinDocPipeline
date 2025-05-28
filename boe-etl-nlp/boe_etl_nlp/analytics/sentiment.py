#!/usr/bin/env python3
"""
Sentiment Analysis Module
========================

This module provides financial sentiment analysis capabilities for the BoE ETL NLP extension.
It includes specialized sentiment analysis for financial text and earnings calls.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging


class SentimentAnalyzer:
    """
    Financial sentiment analyzer for earnings calls and financial documents.
    
    This class provides sentiment analysis specifically tuned for financial language,
    including handling of financial jargon and context-aware sentiment scoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Financial sentiment lexicon (basic implementation)
        self.positive_terms = {
            'growth', 'increase', 'strong', 'positive', 'improved', 'better',
            'exceed', 'outperform', 'robust', 'solid', 'healthy', 'optimistic'
        }
        
        self.negative_terms = {
            'decline', 'decrease', 'weak', 'negative', 'worse', 'deteriorate',
            'underperform', 'challenging', 'difficult', 'concern', 'risk', 'loss'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores and classification
        """
        if not text:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'positive_terms': [],
                'negative_terms': []
            }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count positive and negative terms
        positive_matches = [word for word in words if word in self.positive_terms]
        negative_matches = [word for word in words if word in self.negative_terms]
        
        # Calculate basic sentiment score
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
            confidence = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / len(words)
            confidence = total_sentiment_words / len(words)
            
            if sentiment_score > 0.01:
                sentiment_label = 'positive'
            elif sentiment_score < -0.01:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'positive_terms': positive_matches,
            'negative_terms': negative_matches
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]


def analyze_sentiment(text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze sentiment of a single text.
    
    Args:
        text: Input text to analyze
        config: Optional configuration dictionary
        
    Returns:
        Sentiment analysis results
    """
    analyzer = SentimentAnalyzer(config)
    return analyzer.analyze_sentiment(text)