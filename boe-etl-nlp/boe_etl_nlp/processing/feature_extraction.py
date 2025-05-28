#!/usr/bin/env python3
"""
NLP Feature Extraction Module
=============================

This module provides the main NLPProcessor class that adds comprehensive
NLP features to ETL-processed data. It extracts financial terms, figures,
sentiment, topics, and other linguistic features.

Extracted from the original standalone_frontend.py add_nlp_features() method
and enhanced for modular use.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
import logging


class NLPProcessor:
    """
    Main NLP processor for adding comprehensive NLP features to financial data.
    
    This class provides methods to extract financial terms, figures, sentiment,
    topics, and other linguistic features from text data processed by the core ETL.
    
    Example:
        >>> from boe_etl_nlp import NLPProcessor
        >>> processor = NLPProcessor()
        >>> enhanced_df = processor.add_nlp_features(df)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the NLP processor.
        
        Args:
            config: Optional configuration dictionary for customizing behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load financial terms vocabulary
        self.financial_terms_list = self._load_financial_terms()
        
    def _load_financial_terms(self) -> List[str]:
        """Load comprehensive financial terms list."""
        return [
            'revenue', 'income', 'profit', 'earnings', 'billion', 'million',
            'eps', 'capital', 'assets', 'growth', 'performance', 'margin',
            'return', 'yield', 'dividend', 'interest', 'loan', 'credit',
            'deposit', 'fee', 'commission', 'expense', 'cost', 'investment',
            'portfolio', 'risk', 'regulatory', 'compliance', 'basel',
            'tier', 'ratio', 'liquidity', 'solvency', 'provision', 'reserves',
            'allowance', 'writeoff', 'impairment', 'goodwill', 'intangible',
            'tangible', 'book', 'market', 'fair', 'value', 'valuation',
            'acquisition', 'merger', 'divestiture', 'spinoff', 'ipo',
            'buyback', 'repurchase', 'issuance', 'debt', 'equity', 'bond',
            'note', 'facility', 'line', 'commitment', 'exposure', 'concentration'
        ]
    
    def add_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive NLP features with proper missing value handling.
        
        This method adds 25+ NLP features including financial term extraction,
        figure detection, sentiment analysis, topic classification, and more.
        
        Args:
            df: Input DataFrame with basic ETL fields (text, speaker_norm, etc.)
            
        Returns:
            Enhanced DataFrame with additional NLP features
        """
        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # Handle missing text values first
        enhanced_df['text'] = enhanced_df['text'].fillna('').astype(str)
        enhanced_df['speaker_norm'] = enhanced_df['speaker_norm'].fillna('UNKNOWN').astype(str)
        
        # Basic text metrics (with zero defaults for missing)
        enhanced_df['word_count'] = enhanced_df['text'].str.split().str.len().fillna(0).astype(int)
        enhanced_df['char_count'] = enhanced_df['text'].str.len().fillna(0).astype(int)
        
        # Extract financial terms from each text (never null)
        enhanced_df['all_financial_terms'] = enhanced_df['text'].apply(self._extract_financial_terms)
        
        # Extract financial figures (numbers with financial context) - never null
        enhanced_df['financial_figures'] = enhanced_df['text'].apply(self._extract_financial_figures)
        enhanced_df['financial_figures_text'] = enhanced_df['financial_figures']  # Compatibility
        
        # Classify actual vs projected financial data (never null)
        enhanced_df['data_type'] = enhanced_df['text'].apply(self._classify_actual_vs_projection)
        
        # Boolean flags for data type (never null)
        enhanced_df['is_actual_data'] = (enhanced_df['data_type'] == 'actual').astype(bool)
        enhanced_df['is_projection_data'] = (enhanced_df['data_type'] == 'projection').astype(bool)
        enhanced_df['is_unclear_data'] = (enhanced_df['data_type'] == 'unclear').astype(bool)
        enhanced_df['is_unknown_data'] = (enhanced_df['data_type'] == 'unknown').astype(bool)
        
        # Financial content detection (enhanced) - never null
        enhanced_df['is_financial_content'] = (
            (enhanced_df['all_financial_terms'] != 'NONE') |
            (enhanced_df['financial_figures'] != 'NONE')
        ).astype(bool)
        
        # Speaker analysis (never null)
        enhanced_df['is_management'] = enhanced_df['speaker_norm'].str.contains(
            'CEO|CFO|Chief', case=False, na=False
        ).astype(bool)
        enhanced_df['is_analyst'] = enhanced_df['speaker_norm'].str.contains(
            'Analyst', case=False, na=False
        ).astype(bool)
        enhanced_df['is_named_speaker'] = (enhanced_df['speaker_norm'] != 'UNKNOWN').astype(bool)
        
        # Enhanced topic assignment (never null)
        enhanced_df['primary_topic'] = enhanced_df['text'].apply(self._assign_topic)
        
        # Enhanced topic flags (never null)
        enhanced_df['has_financial_topic'] = enhanced_df['primary_topic'].str.contains(
            'Revenue|Capital|Risk', na=False
        ).astype(bool)
        enhanced_df['has_strategy_topic'] = enhanced_df['primary_topic'].str.contains(
            'Strategy', na=False
        ).astype(bool)
        enhanced_df['has_operational_topic'] = enhanced_df['primary_topic'].str.contains(
            'Operational', na=False
        ).astype(bool)
        enhanced_df['has_unknown_topic'] = (enhanced_df['primary_topic'] == 'Unknown').astype(bool)
        
        # Temporal indicators
        enhanced_df['temporal_indicators'] = enhanced_df['text'].apply(self._extract_temporal_indicators)
        enhanced_df['has_temporal_language'] = (enhanced_df['temporal_indicators'] != 'NONE').astype(bool)
        
        # Additional financial flags
        enhanced_df['has_financial_terms'] = (enhanced_df['all_financial_terms'] != 'NONE').astype(bool)
        enhanced_df['has_financial_figures'] = (enhanced_df['financial_figures'] != 'NONE').astype(bool)
        
        # Processing metadata
        enhanced_df['processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Final missing value check and cleanup
        self._ensure_no_missing_values(enhanced_df)
        
        return enhanced_df
    
    def _extract_financial_terms(self, text: str) -> str:
        """Extract financial terms from text."""
        if pd.isna(text) or text == '':
            return 'NONE'
        
        text_lower = str(text).lower()
        found_terms = []
        for term in self.financial_terms_list:
            if term in text_lower:
                found_terms.append(term)
        return '|'.join(found_terms) if found_terms else 'NONE'
    
    def _extract_financial_figures(self, text: str) -> str:
        """Extract financial figures (numbers with financial context)."""
        if pd.isna(text) or text == '':
            return 'NONE'
        
        text_str = str(text)
        
        # Patterns for financial figures
        patterns = [
            r'\$[\d,]+\.?\d*\s*(?:billion|million|thousand|B|M|K)?',  # Dollar amounts
            r'[\d,]+\.?\d*\s*(?:billion|million|thousand|percent|%|basis points|bps)',  # Numbers with units
            r'[\d,]+\.?\d*\s*(?:dollars|cents)',  # Dollar/cent amounts
            r'(?:approximately|about|around|roughly)\s*[\d,]+\.?\d*',  # Approximate figures
        ]
        
        figures = []
        for pattern in patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            figures.extend(matches)
        
        return '|'.join(figures) if figures else 'NONE'
    
    def _classify_actual_vs_projection(self, text: str) -> str:
        """Classify text as actual vs projected financial data."""
        if pd.isna(text) or text == '':
            return 'unknown'
        
        text_lower = str(text).lower()
        
        # Projection indicators
        projection_terms = [
            'expect', 'forecast', 'project', 'anticipate', 'estimate',
            'guidance', 'outlook', 'target', 'goal', 'plan', 'intend',
            'will be', 'should be', 'likely to', 'going forward',
            'next quarter', 'next year', 'future', 'upcoming'
        ]
        
        # Actual/historical indicators
        actual_terms = [
            'reported', 'achieved', 'delivered', 'recorded', 'posted',
            'was', 'were', 'had', 'generated', 'earned', 'realized',
            'last quarter', 'previous', 'year-over-year', 'compared to'
        ]
        
        projection_score = sum(1 for term in projection_terms if term in text_lower)
        actual_score = sum(1 for term in actual_terms if term in text_lower)
        
        if projection_score > actual_score:
            return 'projection'
        elif actual_score > projection_score:
            return 'actual'
        else:
            return 'unclear'
    
    def _assign_topic(self, text: str) -> str:
        """Assign primary topic to text."""
        if pd.isna(text) or text == '':
            return 'Unknown'
        
        text_lower = str(text).lower()
        if any(term in text_lower for term in ['revenue', 'income', 'growth', 'earnings']):
            return 'Revenue & Growth'
        elif any(term in text_lower for term in ['risk', 'credit', 'provision', 'loss']):
            return 'Risk Management'
        elif any(term in text_lower for term in ['capital', 'regulatory', 'basel', 'tier']):
            return 'Capital & Regulatory'
        elif any(term in text_lower for term in ['strategy', 'outlook', 'guidance', 'plan']):
            return 'Strategy & Outlook'
        elif any(term in text_lower for term in ['cost', 'efficiency', 'expense', 'margin']):
            return 'Operational Efficiency'
        elif any(term in text_lower for term in ['digital', 'technology', 'innovation']):
            return 'Digital & Technology'
        else:
            return 'General Banking'
    
    def _extract_temporal_indicators(self, text: str) -> str:
        """Extract temporal indicators from text."""
        if pd.isna(text) or text == '':
            return 'NONE'
        
        text_lower = str(text).lower()
        temporal_terms = [
            'quarter', 'year', 'month', 'annual', 'quarterly', 'monthly',
            'q1', 'q2', 'q3', 'q4', 'fy', 'ytd', 'mtd', 'qoq', 'yoy',
            'previous', 'prior', 'last', 'next', 'future', 'upcoming',
            'historical', 'current', 'recent', 'latest'
        ]
        
        found_terms = []
        for term in temporal_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return '|'.join(found_terms) if found_terms else 'NONE'
    
    def _ensure_no_missing_values(self, df: pd.DataFrame) -> None:
        """Ensure no missing values in the dataset for NLP compatibility."""
        
        # Define default values for different column types
        string_defaults = {
            'all_financial_terms': 'NONE',
            'financial_figures': 'NONE',
            'financial_figures_text': 'NONE',
            'data_type': 'unknown',
            'primary_topic': 'Unknown',
            'speaker_norm': 'UNKNOWN',
            'text': '',
            'source_file': 'unknown.txt',
            'institution': 'Unknown',
            'quarter': 'Unknown',
            'temporal_indicators': 'NONE',
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        numeric_defaults = {
            'word_count': 0,
            'char_count': 0,
            'sentence_id': 0
        }
        
        boolean_defaults = {
            'is_actual_data': False,
            'is_projection_data': False,
            'is_unclear_data': False,
            'is_unknown_data': True,
            'is_financial_content': False,
            'is_management': False,
            'is_analyst': False,
            'is_named_speaker': False,
            'has_financial_topic': False,
            'has_strategy_topic': False,
            'has_operational_topic': False,
            'has_unknown_topic': True,
            'has_temporal_language': False,
            'has_financial_terms': False,
            'has_financial_figures': False
        }
        
        # Apply defaults for each column type
        for col, default_val in string_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val).astype(str)
        
        for col, default_val in numeric_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val).astype(int)
        
        for col, default_val in boolean_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val).astype(bool)


def add_nlp_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to add NLP features to a DataFrame.
    
    Args:
        df: Input DataFrame with basic ETL fields
        config: Optional configuration dictionary
        
    Returns:
        Enhanced DataFrame with NLP features
    """
    processor = NLPProcessor(config)
    return processor.add_nlp_features(df)