"""
Enhanced Hybrid Topic Modeling Engine for Financial Risk Monitoring
Combines seed-based classification with BERTopic emergent discovery
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class HybridTopicEngine:
    """
    Advanced hybrid topic modeling engine combining seed-based and emergent discovery
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.seed_topics = self._load_seed_topics()
        
        # Topic modeling configuration
        self.seed_threshold = self.config.get('topic_modeling', {}).get('seed_threshold', 3)
        self.min_cluster_size = self.config.get('topic_modeling', {}).get('emergent_min_cluster_size', 50)
        self.coherence_threshold = self.config.get('topic_modeling', {}).get('coherence_threshold', 0.4)
        self.max_topics = self.config.get('topic_modeling', {}).get('max_topics', 50)
        
        # Initialize embedding model
        embedding_model_name = self.config.get('topic_modeling', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize BERTopic model
        self.bertopic_model = None
        self._setup_bertopic()
        
        # Statistics tracking
        self.processing_stats = {
            'seed_assigned': 0,
            'emergent_assigned': 0,
            'miscellaneous': 0,
            'total_processed': 0
        }
        
        logging.info(f"Hybrid topic engine initialized with {len(self.seed_topics)} seed topics")
    
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
    
    def _load_seed_topics(self) -> Dict:
        """Load seed topics from configuration"""
        try:
            seed_topics_path = Path(__file__).parent.parent.parent / "config" / "seed_topics.yaml"
            with open(seed_topics_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('seed_topics', {})
        except FileNotFoundError:
            logging.warning("Seed topics config not found. Using default topics.")
            return self._get_default_seed_topics()
    
    def _get_default_seed_topics(self) -> Dict:
        """Return default seed topics if config is not available"""
        return {
            'credit_risk': {
                'keywords': {
                    'primary': ['loan', 'credit', 'default', 'provision'],
                    'secondary': ['borrower', 'underwriting', 'recovery']
                },
                'weight': 1.0,
                'min_confidence': 0.7
            },
            'operational_risk': {
                'keywords': {
                    'primary': ['cyber', 'fraud', 'compliance', 'operational'],
                    'secondary': ['incident', 'failure', 'disruption']
                },
                'weight': 1.0,
                'min_confidence': 0.7
            }
        }
    
    def _setup_bertopic(self):
        """Setup BERTopic model with optimized parameters"""
        # Vectorizer for topic representation
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=1000,
            lowercase=True
        )
        
        # Initialize BERTopic
        self.bertopic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=self.min_cluster_size,
            nr_topics=self.max_topics,
            calculate_probabilities=True,
            verbose=False
        )
    
    def assign_seed_topics(self, sentences: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assign seed topics based on keyword matching with confidence scoring
        
        Args:
            sentences: DataFrame with text data
            
        Returns:
            Tuple of (seed_assigned_df, remaining_df)
        """
        seed_assigned = []
        remaining = []
        
        for idx, row in sentences.iterrows():
            text = str(row.get('text', '')).lower()
            best_topic, best_score, best_confidence = self._calculate_best_seed_topic(text)
            
            if best_topic and best_confidence >= self.seed_topics[best_topic]['min_confidence']:
                row_copy = row.copy()
                row_copy['topic_seed'] = best_topic
                row_copy['topic_confidence'] = best_confidence
                row_copy['topic_score'] = best_score
                row_copy['topic_type'] = 'seed'
                seed_assigned.append(row_copy)
                self.processing_stats['seed_assigned'] += 1
            else:
                remaining.append(row)
        
        seed_df = pd.DataFrame(seed_assigned) if seed_assigned else pd.DataFrame()
        remaining_df = pd.DataFrame(remaining) if remaining else pd.DataFrame()
        
        logging.info(f"Seed topic assignment: {len(seed_assigned)} assigned, {len(remaining)} remaining")
        
        return seed_df, remaining_df
    
    def _calculate_best_seed_topic(self, text: str) -> Tuple[Optional[str], float, float]:
        """
        Calculate the best matching seed topic for a text
        
        Args:
            text: Input text (already lowercased)
            
        Returns:
            Tuple of (topic_name, raw_score, confidence)
        """
        if not text.strip():
            return None, 0.0, 0.0
        
        # Tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        word_set = set(words)
        
        best_topic = None
        best_score = 0
        best_confidence = 0
        
        for topic_name, topic_config in self.seed_topics.items():
            # Calculate keyword matches
            primary_keywords = topic_config['keywords']['primary']
            secondary_keywords = topic_config['keywords']['secondary']
            
            primary_matches = sum(1 for kw in primary_keywords if kw in word_set)
            secondary_matches = sum(1 for kw in secondary_keywords if kw in word_set)
            
            # Weighted score calculation
            raw_score = primary_matches * 2.0 + secondary_matches * 1.0
            
            # Confidence calculation (normalized by text length and keyword coverage)
            total_keywords = len(primary_keywords) + len(secondary_keywords)
            keyword_coverage = (primary_matches + secondary_matches) / total_keywords
            text_relevance = min(raw_score / len(words), 1.0) if words else 0
            
            confidence = (keyword_coverage * 0.6 + text_relevance * 0.4) * topic_config['weight']
            
            if raw_score >= self.seed_threshold and confidence > best_confidence:
                best_topic = topic_name
                best_score = raw_score
                best_confidence = confidence
        
        return best_topic, best_score, best_confidence
    
    def discover_emergent_topics(self, sentences: pd.DataFrame) -> pd.DataFrame:
        """
        Use BERTopic to discover emergent topics in remaining sentences
        
        Args:
            sentences: DataFrame with remaining sentences after seed assignment
            
        Returns:
            DataFrame with emergent topic assignments
        """
        if len(sentences) < self.min_cluster_size:
            # Not enough data for clustering
            sentences = sentences.copy()
            sentences['topic_emergent'] = 'miscellaneous'
            sentences['topic_confidence'] = 0.0
            sentences['topic_type'] = 'misc'
            self.processing_stats['miscellaneous'] += len(sentences)
            logging.info(f"Insufficient data for emergent topics: {len(sentences)} sentences marked as miscellaneous")
            return sentences
        
        # Prepare texts for BERTopic
        texts = sentences['text'].fillna('').astype(str).tolist()
        
        try:
            # Fit and transform
            topics, probabilities = self.bertopic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = self.bertopic_model.get_topic_info()
            
            # Validate topic coherence
            coherence_scores = self._calculate_topic_coherence()
            
            # Assign results
            sentences = sentences.copy()
            for i, (topic_id, prob) in enumerate(zip(topics, probabilities)):
                if topic_id == -1:  # Outlier
                    sentences.iloc[i, sentences.columns.get_loc('topic_emergent')] = 'miscellaneous'
                    sentences.iloc[i, sentences.columns.get_loc('topic_confidence')] = 0.0
                    sentences.iloc[i, sentences.columns.get_loc('topic_type')] = 'misc'
                    self.processing_stats['miscellaneous'] += 1
                else:
                    # Check coherence threshold
                    topic_coherence = coherence_scores.get(topic_id, 0.0)
                    if topic_coherence >= self.coherence_threshold:
                        sentences.iloc[i, sentences.columns.get_loc('topic_emergent')] = f"emergent_{topic_id}"
                        sentences.iloc[i, sentences.columns.get_loc('topic_confidence')] = float(max(prob)) if prob is not None else 0.5
                        sentences.iloc[i, sentences.columns.get_loc('topic_type')] = 'emergent'
                        self.processing_stats['emergent_assigned'] += 1
                    else:
                        sentences.iloc[i, sentences.columns.get_loc('topic_emergent')] = 'miscellaneous'
                        sentences.iloc[i, sentences.columns.get_loc('topic_confidence')] = 0.0
                        sentences.iloc[i, sentences.columns.get_loc('topic_type')] = 'misc'
                        self.processing_stats['miscellaneous'] += 1
            
            # Add topic keywords
            self._add_topic_keywords(sentences, topic_info)
            
            logging.info(f"Emergent topic discovery: {self.processing_stats['emergent_assigned']} assigned, "
                        f"{self.processing_stats['miscellaneous']} miscellaneous")
            
        except Exception as e:
            logging.error(f"Error in emergent topic discovery: {e}")
            # Fallback: mark all as miscellaneous
            sentences = sentences.copy()
            sentences['topic_emergent'] = 'miscellaneous'
            sentences['topic_confidence'] = 0.0
            sentences['topic_type'] = 'misc'
            self.processing_stats['miscellaneous'] += len(sentences)
        
        return sentences
    
    def _calculate_topic_coherence(self) -> Dict[int, float]:
        """
        Calculate coherence scores for discovered topics
        
        Returns:
            Dictionary mapping topic_id to coherence score
        """
        coherence_scores = {}
        
        try:
            if self.bertopic_model is None:
                return coherence_scores
            
            # Get topics
            topics = self.bertopic_model.get_topics()
            
            # Calculate coherence for each topic
            for topic_id in topics.keys():
                if topic_id != -1:  # Skip outlier topic
                    # Simple coherence approximation based on topic word probabilities
                    topic_words = self.bertopic_model.get_topic(topic_id)
                    if topic_words:
                        # Use the average probability as a proxy for coherence
                        coherence = np.mean([prob for word, prob in topic_words[:10]])
                        coherence_scores[topic_id] = coherence
                    else:
                        coherence_scores[topic_id] = 0.0
        
        except Exception as e:
            logging.warning(f"Error calculating coherence: {e}")
        
        return coherence_scores
    
    def _add_topic_keywords(self, sentences: pd.DataFrame, topic_info: pd.DataFrame):
        """Add topic keywords to sentences DataFrame"""
        try:
            # Create topic keyword mapping
            topic_keywords = {}
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:
                    keywords = row.get('Representation', '')
                    topic_keywords[f"emergent_{topic_id}"] = keywords
            
            # Add keywords column
            sentences['topic_keywords'] = sentences['topic_emergent'].map(topic_keywords).fillna('')
            
        except Exception as e:
            logging.warning(f"Error adding topic keywords: {e}")
            sentences['topic_keywords'] = ''
    
    def process_quarter_data(self, bank: str, quarter: str, sentences: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main processing function for a quarter's data
        
        Args:
            bank: Bank name
            quarter: Quarter identifier
            sentences: DataFrame with sentence-level data
            
        Returns:
            Tuple of (processed_df, processing_metadata)
        """
        logging.info(f"Processing {bank} {quarter}: {len(sentences)} sentences")
        
        # Reset statistics
        self.processing_stats = {k: 0 for k in self.processing_stats.keys()}
        self.processing_stats['total_processed'] = len(sentences)
        
        # Ensure required columns exist
        if 'topic_seed' not in sentences.columns:
            sentences['topic_seed'] = None
        if 'topic_emergent' not in sentences.columns:
            sentences['topic_emergent'] = None
        if 'topic_confidence' not in sentences.columns:
            sentences['topic_confidence'] = 0.0
        if 'topic_type' not in sentences.columns:
            sentences['topic_type'] = None
        
        # Step 1: Assign seed topics
        seed_assigned, remaining = self.assign_seed_topics(sentences)
        
        # Step 2: Discover emergent topics in remaining sentences
        if len(remaining) > 0:
            emergent_assigned = self.discover_emergent_topics(remaining)
        else:
            emergent_assigned = pd.DataFrame()
        
        # Step 3: Combine results
        if len(seed_assigned) > 0 and len(emergent_assigned) > 0:
            result = pd.concat([seed_assigned, emergent_assigned], ignore_index=True)
        elif len(seed_assigned) > 0:
            result = seed_assigned
        elif len(emergent_assigned) > 0:
            result = emergent_assigned
        else:
            result = sentences.copy()
            result['topic_seed'] = None
            result['topic_emergent'] = 'miscellaneous'
            result['topic_confidence'] = 0.0
            result['topic_type'] = 'misc'
        
        # Step 4: Add final topic assignment
        result['final_topic'] = result.apply(self._get_final_topic, axis=1)
        
        # Step 5: Add metadata
        result['bank'] = bank
        result['quarter'] = quarter
        result['processing_timestamp'] = pd.Timestamp.now()
        
        # Step 6: Generate processing metadata
        metadata = self._generate_processing_metadata(bank, quarter)
        
        logging.info(f"Processing complete: {len(result)} sentences processed")
        
        return result, metadata
    
    def _get_final_topic(self, row) -> str:
        """Determine final topic assignment for a row"""
        if pd.notna(row.get('topic_seed')):
            return row['topic_seed']
        elif pd.notna(row.get('topic_emergent')):
            return row['topic_emergent']
        else:
            return 'miscellaneous'
    
    def _generate_processing_metadata(self, bank: str, quarter: str) -> Dict:
        """Generate metadata about the processing run"""
        coherence_scores = self._calculate_topic_coherence() if self.bertopic_model else {}
        
        metadata = {
            'bank': bank,
            'quarter': quarter,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'statistics': self.processing_stats.copy(),
            'seed_topics_used': list(self.seed_topics.keys()),
            'emergent_topics_discovered': len(coherence_scores),
            'average_coherence': np.mean(list(coherence_scores.values())) if coherence_scores else 0.0,
            'coherence_scores': coherence_scores,
            'configuration': {
                'seed_threshold': self.seed_threshold,
                'min_cluster_size': self.min_cluster_size,
                'coherence_threshold': self.coherence_threshold,
                'max_topics': self.max_topics
            }
        }
        
        return metadata
    
    def get_topic_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary of topic modeling results
        
        Args:
            df: DataFrame with topic modeling results
            
        Returns:
            Dictionary with topic summary statistics
        """
        summary = {
            'total_sentences': len(df),
            'seed_topic_distribution': df[df['topic_type'] == 'seed']['final_topic'].value_counts().to_dict(),
            'emergent_topic_distribution': df[df['topic_type'] == 'emergent']['final_topic'].value_counts().to_dict(),
            'miscellaneous_count': len(df[df['topic_type'] == 'misc']),
            'average_confidence': df['topic_confidence'].mean(),
            'high_confidence_count': len(df[df['topic_confidence'] > 0.8]),
            'processing_statistics': self.processing_stats.copy()
        }
        
        return summary

def get_hybrid_topic_engine(config_path: Optional[str] = None) -> HybridTopicEngine:
    """Get hybrid topic engine instance"""
    return HybridTopicEngine(config_path)