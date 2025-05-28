#!/usr/bin/env python3
"""
Financial Classification Module
==============================

This module provides financial data classification capabilities for the BoE ETL NLP extension.
It includes classification of financial statements, document types, and content categories.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging


class FinancialClassifier:
    """
    Financial data classifier for earnings calls and financial documents.
    
    This class provides classification capabilities for financial content,
    including document type detection and content categorization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the financial classifier.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Document type classification patterns
        self.document_patterns = {
            'earnings_call': [
                'earnings call', 'quarterly call', 'conference call',
                'q1 call', 'q2 call', 'q3 call', 'q4 call'
            ],
            'earnings_presentation': [
                'presentation', 'slide', 'deck', 'investor presentation'
            ],
            'financial_supplement': [
                'supplement', 'financial data', 'metrics', 'key figures'
            ],
            'financial_report': [
                'annual report', 'quarterly report', '10-k', '10-q',
                'financial statements'
            ],
            'press_release': [
                'press release', 'announcement', 'news release'
            ]
        }
        
        # Content category patterns
        self.content_patterns = {
            'financial_performance': [
                'revenue', 'earnings', 'profit', 'income', 'performance',
                'results', 'financial results'
            ],
            'guidance_outlook': [
                'guidance', 'outlook', 'forecast', 'expectations',
                'projections', 'targets'
            ],
            'risk_factors': [
                'risk', 'risks', 'risk factors', 'uncertainties',
                'challenges', 'headwinds'
            ],
            'regulatory_compliance': [
                'regulatory', 'compliance', 'regulation', 'basel',
                'capital requirements', 'stress test'
            ],
            'business_strategy': [
                'strategy', 'strategic', 'initiatives', 'transformation',
                'digital transformation', 'growth strategy'
            ]
        }
    
    def classify_document_type(self, text: str, filename: str = "") -> Dict[str, Any]:
        """
        Classify document type based on content and filename.
        
        Args:
            text: Document text content
            filename: Optional filename for additional context
            
        Returns:
            Dictionary with classification results
        """
        if not text and not filename:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'matched_patterns': []
            }
        
        combined_text = f"{text} {filename}".lower()
        scores = {}
        matched_patterns = {}
        
        # Score each document type
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            matches = []
            for pattern in patterns:
                if pattern in combined_text:
                    score += 1
                    matches.append(pattern)
            scores[doc_type] = score
            matched_patterns[doc_type] = matches
        
        # Find best match
        if max(scores.values()) == 0:
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'matched_patterns': []
            }
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / len(self.document_patterns[best_type])
        
        return {
            'document_type': best_type,
            'confidence': confidence,
            'matched_patterns': matched_patterns[best_type]
        }
    
    def classify_content_category(self, text: str) -> Dict[str, Any]:
        """
        Classify content category based on text content.
        
        Args:
            text: Text content to classify
            
        Returns:
            Dictionary with classification results
        """
        if not text:
            return {
                'primary_category': 'unknown',
                'category_scores': {},
                'confidence': 0.0
            }
        
        text_lower = text.lower()
        scores = {}
        
        # Score each content category
        for category, patterns in self.content_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            scores[category] = score
        
        # Find primary category
        if max(scores.values()) == 0:
            return {
                'primary_category': 'general',
                'category_scores': scores,
                'confidence': 0.0
            }
        
        primary_category = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = scores[primary_category] / total_matches if total_matches > 0 else 0.0
        
        return {
            'primary_category': primary_category,
            'category_scores': scores,
            'confidence': confidence
        }
    
    def classify_financial_statement_type(self, text: str) -> Dict[str, Any]:
        """
        Classify type of financial statement or data.
        
        Args:
            text: Text content to classify
            
        Returns:
            Dictionary with classification results
        """
        if not text:
            return {
                'statement_type': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
        
        text_lower = text.lower()
        
        # Statement type indicators
        statement_indicators = {
            'income_statement': [
                'revenue', 'income', 'earnings', 'profit', 'loss',
                'operating income', 'net income'
            ],
            'balance_sheet': [
                'assets', 'liabilities', 'equity', 'balance sheet',
                'total assets', 'shareholders equity'
            ],
            'cash_flow': [
                'cash flow', 'operating cash flow', 'free cash flow',
                'cash and cash equivalents'
            ],
            'capital_adequacy': [
                'capital ratio', 'tier 1', 'tier 2', 'risk weighted assets',
                'capital adequacy', 'basel'
            ]
        }
        
        scores = {}
        found_indicators = {}
        
        for stmt_type, indicators in statement_indicators.items():
            matches = [ind for ind in indicators if ind in text_lower]
            scores[stmt_type] = len(matches)
            found_indicators[stmt_type] = matches
        
        if max(scores.values()) == 0:
            return {
                'statement_type': 'general_financial',
                'confidence': 0.0,
                'indicators': []
            }
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / len(statement_indicators[best_type])
        
        return {
            'statement_type': best_type,
            'confidence': confidence,
            'indicators': found_indicators[best_type]
        }
    
    def classify_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of records.
        
        Args:
            records: List of records with text content
            
        Returns:
            List of records with classification results
        """
        classified_records = []
        
        for record in records:
            text = record.get('text', '')
            filename = record.get('source_file', '')
            
            # Perform all classifications
            doc_type = self.classify_document_type(text, filename)
            content_category = self.classify_content_category(text)
            statement_type = self.classify_financial_statement_type(text)
            
            # Add classification results to record
            enhanced_record = record.copy()
            enhanced_record.update({
                'document_type': doc_type['document_type'],
                'document_confidence': doc_type['confidence'],
                'content_category': content_category['primary_category'],
                'content_confidence': content_category['confidence'],
                'statement_type': statement_type['statement_type'],
                'statement_confidence': statement_type['confidence'],
                'classification_metadata': {
                    'doc_type_patterns': doc_type['matched_patterns'],
                    'category_scores': content_category['category_scores'],
                    'statement_indicators': statement_type['indicators']
                }
            })
            
            classified_records.append(enhanced_record)
        
        return classified_records


def classify_document_type(text: str, filename: str = "", config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to classify document type.
    
    Args:
        text: Document text content
        filename: Optional filename
        config: Optional configuration dictionary
        
    Returns:
        Classification results
    """
    classifier = FinancialClassifier(config)
    return classifier.classify_document_type(text, filename)