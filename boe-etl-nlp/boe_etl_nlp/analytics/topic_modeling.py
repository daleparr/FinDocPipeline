#!/usr/bin/env python3
"""
Topic Modeling Module
====================

This module provides advanced topic modeling capabilities using a hybrid approach
that combines seed themes with BERTopic for emerging topic discovery.

Migrated from boe-etl package and adapted for the NLP extension.
"""

import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import logging

# Optional imports for advanced features
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logging.warning("BERTopic not available. Advanced topic modeling will be disabled.")

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Using basic tokenization.")


class TopicModeler:
    """
    Hybrid topic modeling class that combines seed themes with advanced topic discovery.
    
    This class provides both rule-based topic assignment using predefined themes
    and machine learning-based topic discovery using BERTopic (if available).
    
    Example:
        >>> from boe_etl_nlp import TopicModeler
        >>> modeler = TopicModeler()
        >>> results = modeler.process_batch(records, 'JPMorgan', 'Q1_2025')
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the topic modeler.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.seed_themes = self._load_seed_themes()
        self.vectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=0.01,
            max_df=0.95
        )
        
        # Initialize BERTopic if available
        if BERTOPIC_AVAILABLE:
            try:
                self.bertopic = BERTopic(
                    embedding_model="ProsusAI/finbert",
                    language="english",
                    calculate_probabilities=True,
                    verbose=False
                )
            except Exception as e:
                logging.warning(f"Failed to initialize BERTopic with FinBERT: {e}")
                self.bertopic = BERTopic(language="english", verbose=False)
        else:
            self.bertopic = None
        
        self.logger = logging.getLogger(__name__)
        
    def _load_seed_themes(self) -> Dict[str, List[str]]:
        """Load seed themes from configuration or use defaults."""
        # Default seed themes for financial documents
        default_themes = {
            "Revenue & Growth": [
                "revenue", "income", "growth", "earnings", "sales", "profit",
                "margin", "performance", "increase", "expansion", "organic"
            ],
            "Risk Management": [
                "risk", "credit", "provision", "loss", "allowance", "impairment",
                "default", "exposure", "concentration", "mitigation"
            ],
            "Capital & Regulatory": [
                "capital", "regulatory", "basel", "tier", "ratio", "compliance",
                "requirement", "buffer", "adequacy", "leverage"
            ],
            "Strategy & Outlook": [
                "strategy", "outlook", "guidance", "plan", "target", "goal",
                "initiative", "transformation", "future", "vision"
            ],
            "Operational Efficiency": [
                "cost", "efficiency", "expense", "operational", "productivity",
                "automation", "optimization", "streamline", "reduce"
            ],
            "Digital & Technology": [
                "digital", "technology", "innovation", "platform", "data",
                "analytics", "artificial", "intelligence", "automation", "cyber"
            ]
        }
        
        # Try to load from config file if specified
        if "seed_themes_path" in self.config:
            try:
                theme_path = Path(self.config["seed_themes_path"])
                with open(theme_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logging.warning(f"Failed to load seed themes from {theme_path}: {e}")
        
        return default_themes
    
    def _preprocess_text(self, text: str) -> str:
        """Text preprocessing for topic modeling."""
        if not text:
            return ""
        
        text = text.lower()
        
        if NLTK_AVAILABLE:
            # Use NLTK for better tokenization
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
        else:
            # Basic tokenization fallback
            import re
            tokens = re.findall(r'\b[a-zA-Z]+\b', text)
            # Basic stopwords
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [t for t in tokens if t not in stop_words]
        
        return ' '.join(tokens)
    
    def assign_seed_theme(self, text: str, threshold: int = 2) -> Optional[str]:
        """
        Assign text to a seed theme based on keyword matching.
        
        Args:
            text: Input text to classify
            threshold: Minimum keyword matches required
            
        Returns:
            Theme name if match found, None otherwise
        """
        if not text:
            return None
            
        processed_text = self._preprocess_text(text)
        scores = {}
        
        for theme, keywords in self.seed_themes.items():
            score = sum(processed_text.count(keyword) for keyword in keywords)
            scores[theme] = score
        
        # Get best theme
        if scores:
            best_theme, best_score = max(scores.items(), key=lambda kv: kv[1])
            return best_theme if best_score >= threshold else None
        
        return None
    
    def process_seed_themes(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process records using seed themes.
        
        Args:
            records: List of text records to process
            
        Returns:
            Tuple of (seed_assigned_records, unassigned_records)
        """
        seed_assigned = []
        misc_corpus = []
        
        for record in records:
            theme = self.assign_seed_theme(record.get("text", ""))
            if theme:
                record_copy = record.copy()
                record_copy["topic_label"] = theme
                record_copy["topic_confidence"] = 1.0  # High confidence for seed themes
                record_copy["topic_method"] = "seed_theme"
                seed_assigned.append(record_copy)
            else:
                misc_corpus.append(record)
        
        self.logger.info(f"Assigned {len(seed_assigned)} records to seed themes")
        return seed_assigned, misc_corpus
    
    def process_emerging_topics(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process records using BERTopic for emerging topics.
        
        Args:
            records: List of unassigned text records
            
        Returns:
            List of records with topic assignments
        """
        if not records:
            return []
        
        texts = [r.get("text", "") for r in records]
        processed_records = []
        
        # Handle single document case
        if len(texts) == 1:
            self.logger.info("Only one document found - assigning default topic")
            record = records[0].copy()
            record["topic_label"] = "General_0"
            record["topic_confidence"] = 0.5
            record["topic_method"] = "default"
            record["topic_keywords"] = "document, text, content"
            return [record]
        
        # Use BERTopic if available
        if self.bertopic and BERTOPIC_AVAILABLE:
            try:
                # Filter out empty texts
                valid_texts = [t for t in texts if t.strip()]
                if len(valid_texts) < 2:
                    # Fallback for insufficient data
                    for i, record in enumerate(records):
                        record_copy = record.copy()
                        record_copy["topic_label"] = f"General_{i}"
                        record_copy["topic_confidence"] = 0.5
                        record_copy["topic_method"] = "fallback"
                        processed_records.append(record_copy)
                    return processed_records
                
                # Fit BERTopic model
                topics, probs = self.bertopic.fit_transform(valid_texts)
                
                # Get topic representations
                topic_info = self.bertopic.get_topic_info()
                
                # Update records with topic information
                valid_idx = 0
                for record in records:
                    record_copy = record.copy()
                    if record.get("text", "").strip():
                        topic = topics[valid_idx]
                        prob = probs[valid_idx] if probs is not None else 0.5
                        
                        if topic != -1:  # Skip outliers
                            record_copy["topic_label"] = f"Emerging_{topic}"
                            record_copy["topic_confidence"] = float(prob)
                            record_copy["topic_method"] = "bertopic"
                            
                            # Add top keywords
                            topic_row = topic_info[topic_info["Topic"] == topic]
                            if not topic_row.empty:
                                record_copy["topic_keywords"] = topic_row["Representation"].iloc[0]
                        else:
                            record_copy["topic_label"] = "Outlier"
                            record_copy["topic_confidence"] = 0.1
                            record_copy["topic_method"] = "outlier"
                        
                        valid_idx += 1
                    else:
                        record_copy["topic_label"] = "Empty"
                        record_copy["topic_confidence"] = 0.0
                        record_copy["topic_method"] = "empty"
                    
                    processed_records.append(record_copy)
                    
            except Exception as e:
                self.logger.error(f"Error in BERTopic modeling: {e}")
                # Fallback to simple assignment
                for i, record in enumerate(records):
                    record_copy = record.copy()
                    record_copy["topic_label"] = f"Topic_{i}"
                    record_copy["topic_confidence"] = 0.5
                    record_copy["topic_method"] = "fallback"
                    processed_records.append(record_copy)
        else:
            # Simple fallback when BERTopic is not available
            for i, record in enumerate(records):
                record_copy = record.copy()
                record_copy["topic_label"] = f"Topic_{i}"
                record_copy["topic_confidence"] = 0.5
                record_copy["topic_method"] = "simple"
                processed_records.append(record_copy)
        
        self.logger.info(f"Assigned {len(processed_records)} records to emerging topics")
        return processed_records
    
    def analyze_topics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze topic distribution and statistics.
        
        Args:
            records: List of records with topic assignments
            
        Returns:
            Dictionary with topic analysis results
        """
        if not records:
            return {"error": "No records to analyze"}
        
        df = pd.DataFrame(records)
        
        # Topic distribution
        topic_dist = df["topic_label"].value_counts(normalize=True).to_dict()
        
        # Confidence scores
        avg_confidence = df["topic_confidence"].mean() if "topic_confidence" in df.columns else 0.0
        
        # Method distribution
        method_dist = df["topic_method"].value_counts().to_dict() if "topic_method" in df.columns else {}
        
        # Keyword frequency (if available)
        all_text = ' '.join(df["text"].fillna(''))
        if NLTK_AVAILABLE:
            tokens = word_tokenize(all_text.lower())
        else:
            import re
            tokens = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
        
        keyword_freq = Counter(tokens).most_common(20)
        
        return {
            "topic_distribution": topic_dist,
            "average_confidence": avg_confidence,
            "method_distribution": method_dist,
            "top_keywords": keyword_freq,
            "total_records": len(records),
            "seed_theme_count": len(df[df["topic_method"] == "seed_theme"]) if "topic_method" in df.columns else 0,
            "emerging_topic_count": len(df[df["topic_method"] == "bertopic"]) if "topic_method" in df.columns else 0
        }
    
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        bank_name: str,
        quarter: str
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of records through the hybrid topic modeling pipeline.
        
        Args:
            records: List of text records to process
            bank_name: Name of the financial institution
            quarter: Quarter identifier (e.g., 'Q1_2025')
            
        Returns:
            List of records with topic assignments and metadata
        """
        try:
            self.logger.info(f"Processing {len(records)} records for {bank_name} {quarter}")
            
            # Stage 1: Seed theme assignment
            seed_assigned, misc_corpus = self.process_seed_themes(records)
            
            # Stage 2: Emerging topic modeling
            emerging_records = self.process_emerging_topics(misc_corpus)
            
            # Combine results
            all_records = seed_assigned + emerging_records
            
            # Add metadata
            for record in all_records:
                record["bank_name"] = bank_name
                record["quarter"] = quarter
                record["processing_date"] = datetime.now().isoformat()
                record["topic_model_version"] = "1.0.0"
            
            # Analyze results
            analysis = self.analyze_topics(all_records)
            self.logger.info(f"Topic analysis: {analysis}")
            
            return all_records
            
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
            raise


def get_topic_modeler(config: Optional[Dict] = None) -> TopicModeler:
    """
    Get a topic modeler instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TopicModeler instance
    """
    return TopicModeler(config)