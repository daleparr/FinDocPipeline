"""
Advanced FinBERT-based sentiment analysis for financial risk monitoring
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from typing import Dict, List, Tuple, Optional
import logging
import yaml
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinBERTAnalyzer:
    """
    Advanced FinBERT sentiment analyzer with risk-specific features
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model_name = self.config.get('sentiment_analysis', {}).get('model_name', 'ProsusAI/finbert')
        self.batch_size = self.config.get('sentiment_analysis', {}).get('batch_size', 32)
        self.confidence_threshold = self.config.get('sentiment_analysis', {}).get('confidence_threshold', 0.7)
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Create sentiment pipeline
        device = 0 if torch.cuda.is_available() and self.config.get('sentiment_analysis', {}).get('use_gpu', True) else -1
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            return_all_scores=True
        )
        
        # Load risk-specific lexicons
        self._load_risk_lexicons()
        
        logging.info(f"FinBERT analyzer initialized with model: {self.model_name}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "risk_monitoring_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
    
    def _load_risk_lexicons(self):
        """Load risk-specific lexicons from seed topics configuration"""
        try:
            seed_topics_path = Path(__file__).parent.parent.parent / "config" / "seed_topics.yaml"
            with open(seed_topics_path, 'r') as f:
                seed_config = yaml.safe_load(f)
            
            self.risk_escalation = seed_config.get('risk_escalation_patterns', {})
            self.uncertainty_markers = seed_config.get('uncertainty_markers', [])
            self.hedging_language = seed_config.get('hedging_language', [])
            self.formality_indicators = seed_config.get('formality_indicators', {})
            
        except FileNotFoundError:
            logging.warning("Seed topics config not found. Using default lexicons.")
            self._set_default_lexicons()
    
    def _set_default_lexicons(self):
        """Set default lexicons if config file is not available"""
        self.risk_escalation = {
            'deteriorating': ['deteriorating', 'worsening', 'declining', 'challenging'],
            'mitigation': ['improving', 'strengthening', 'robust', 'stable']
        }
        self.uncertainty_markers = ['may', 'might', 'could', 'uncertain', 'potential']
        self.hedging_language = ['approximately', 'roughly', 'around', 'likely', 'probably']
        self.formality_indicators = {
            'high': ['pursuant', 'notwithstanding', 'therefore'],
            'low': ['really', 'pretty', 'basically']
        }
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a list of texts using FinBERT
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[str]) -> List[Dict]:
        """Process a batch of texts through FinBERT"""
        try:
            # Get FinBERT predictions
            predictions = self.sentiment_pipeline(batch)
            
            results = []
            for text, pred in zip(batch, predictions):
                # Extract sentiment scores
                sentiment_scores = {item['label'].lower(): item['score'] for item in pred}
                
                # Determine primary sentiment
                primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
                
                # Calculate additional features
                tone_features = self._analyze_tone(text)
                risk_features = self._analyze_risk_language(text)
                
                result = {
                    'sentiment_label': primary_sentiment[0],
                    'sentiment_score': primary_sentiment[1],
                    'sentiment_positive': sentiment_scores.get('positive', 0.0),
                    'sentiment_negative': sentiment_scores.get('negative', 0.0),
                    'sentiment_neutral': sentiment_scores.get('neutral', 0.0),
                    'sentiment_confidence': primary_sentiment[1],
                    **tone_features,
                    **risk_features
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            # Return default results for the batch
            return [self._get_default_result() for _ in batch]
    
    def _analyze_tone(self, text: str) -> Dict:
        """Analyze tone-related features"""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return self._get_default_tone_features()
        
        # Hedging score
        hedging_count = sum(1 for word in self.hedging_language if word in text_lower)
        hedging_score = hedging_count / word_count
        
        # Uncertainty score
        uncertainty_count = sum(1 for word in self.uncertainty_markers if word in text_lower)
        uncertainty_score = uncertainty_count / word_count
        
        # Formality score
        formal_count = sum(1 for word in self.formality_indicators.get('high', []) if word in text_lower)
        informal_count = sum(1 for word in self.formality_indicators.get('low', []) if word in text_lower)
        formality_score = (formal_count - informal_count) / word_count if word_count > 0 else 0.0
        
        # Complexity score (based on average word length and sentence structure)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sentence_count = len(re.split(r'[.!?]+', text))
        complexity_score = (avg_word_length / 10.0) + (word_count / sentence_count / 20.0) if sentence_count > 0 else 0
        complexity_score = min(complexity_score, 1.0)  # Cap at 1.0
        
        return {
            'hedging_score': hedging_score,
            'uncertainty_score': uncertainty_score,
            'formality_score': formality_score,
            'complexity_score': complexity_score
        }
    
    def _analyze_risk_language(self, text: str) -> Dict:
        """Analyze risk-specific language patterns"""
        text_lower = text.lower()
        
        # Risk escalation language
        deteriorating_count = sum(1 for word in self.risk_escalation.get('deteriorating', []) if word in text_lower)
        mitigation_count = sum(1 for word in self.risk_escalation.get('mitigation', []) if word in text_lower)
        
        # Risk escalation score (positive means more risk escalation language)
        risk_escalation_score = (deteriorating_count - mitigation_count) / len(text_lower.split()) if text_lower.split() else 0
        
        # Financial stress indicators
        stress_indicators = ['stress', 'pressure', 'concern', 'worry', 'issue', 'problem', 'challenge']
        stress_count = sum(1 for indicator in stress_indicators if indicator in text_lower)
        stress_score = stress_count / len(text_lower.split()) if text_lower.split() else 0
        
        # Confidence indicators
        confidence_indicators = ['confident', 'strong', 'solid', 'robust', 'healthy', 'optimistic']
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in text_lower)
        confidence_score = confidence_count / len(text_lower.split()) if text_lower.split() else 0
        
        return {
            'risk_escalation_score': risk_escalation_score,
            'stress_score': stress_score,
            'confidence_score': confidence_score
        }
    
    def _get_default_tone_features(self) -> Dict:
        """Return default tone features for empty or invalid text"""
        return {
            'hedging_score': 0.0,
            'uncertainty_score': 0.0,
            'formality_score': 0.0,
            'complexity_score': 0.0
        }
    
    def _get_default_result(self) -> Dict:
        """Return default result for failed analysis"""
        return {
            'sentiment_label': 'neutral',
            'sentiment_score': 0.33,
            'sentiment_positive': 0.33,
            'sentiment_negative': 0.33,
            'sentiment_neutral': 0.34,
            'sentiment_confidence': 0.33,
            'hedging_score': 0.0,
            'uncertainty_score': 0.0,
            'formality_score': 0.0,
            'complexity_score': 0.0,
            'risk_escalation_score': 0.0,
            'stress_score': 0.0,
            'confidence_score': 0.0
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Add sentiment analysis to a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with sentiment analysis columns added
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Handle missing values
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Analyze sentiment
        sentiment_results = self.analyze_sentiment(texts)
        
        # Add results to DataFrame
        df_copy = df.copy()
        for i, result in enumerate(sentiment_results):
            for key, value in result.items():
                df_copy.loc[i, key] = value
        
        return df_copy
    
    def aggregate_sentiment_by_speaker(self, df: pd.DataFrame, speaker_weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Aggregate sentiment scores by speaker with optional weighting
        
        Args:
            df: DataFrame with sentiment analysis results
            speaker_weights: Optional dictionary of speaker weights
            
        Returns:
            DataFrame with aggregated sentiment by speaker
        """
        if speaker_weights is None:
            speaker_weights = self.config.get('speaker_weights', {})
        
        # Add speaker weights
        df['speaker_weight'] = df['speaker_norm'].map(speaker_weights).fillna(0.2)
        
        # Calculate weighted sentiment scores
        sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
        tone_cols = ['hedging_score', 'uncertainty_score', 'formality_score', 'complexity_score']
        risk_cols = ['risk_escalation_score', 'stress_score', 'confidence_score']
        
        aggregation_dict = {}
        
        # Weighted averages for sentiment and tone features
        for col in sentiment_cols + tone_cols + risk_cols:
            if col in df.columns:
                aggregation_dict[f'weighted_{col}'] = lambda x, col=col: np.average(
                    x[col], weights=x['speaker_weight']
                ) if len(x) > 0 else 0.0
        
        # Count and basic stats
        aggregation_dict.update({
            'sentence_count': 'count',
            'total_weight': lambda x: x['speaker_weight'].sum(),
            'avg_confidence': lambda x: x['sentiment_confidence'].mean()
        })
        
        result = df.groupby('speaker_norm').agg(aggregation_dict).reset_index()
        
        return result
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary of sentiment analysis results
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        summary = {
            'total_sentences': len(df),
            'sentiment_distribution': df['sentiment_label'].value_counts(normalize=True).to_dict(),
            'average_confidence': df['sentiment_confidence'].mean(),
            'average_hedging': df['hedging_score'].mean(),
            'average_uncertainty': df['uncertainty_score'].mean(),
            'average_formality': df['formality_score'].mean(),
            'average_complexity': df['complexity_score'].mean(),
            'average_risk_escalation': df['risk_escalation_score'].mean(),
            'average_stress': df['stress_score'].mean(),
            'average_confidence_score': df['confidence_score'].mean(),
            'high_confidence_sentences': len(df[df['sentiment_confidence'] > self.confidence_threshold]),
            'high_risk_sentences': len(df[df['risk_escalation_score'] > 0.1]),
            'high_uncertainty_sentences': len(df[df['uncertainty_score'] > 0.1])
        }
        
        return summary

def get_finbert_analyzer(config_path: Optional[str] = None) -> FinBERTAnalyzer:
    """Get FinBERT analyzer instance"""
    return FinBERTAnalyzer(config_path)