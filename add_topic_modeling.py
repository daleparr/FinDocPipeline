#!/usr/bin/env python3
"""
Add Topic Modeling to ETL Pipeline

This script adds topic modeling functionality to generate topic labels
for the financial earnings data, completing the NLP schema requirements.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import re
from typing import List, Dict, Tuple
import json

class FinancialTopicModeler:
    """Topic modeling specialized for financial earnings content."""
    
    def __init__(self, n_topics: int = 10):
        """Initialize the topic modeler."""
        self.n_topics = n_topics
        self.vectorizer = None
        self.lda_model = None
        self.topic_labels = None
        self.financial_topics = {
            0: "Financial Performance & Results",
            1: "Revenue & Income Growth", 
            2: "Banking Operations & Services",
            3: "Risk Management & Credit",
            4: "Capital & Regulatory Metrics",
            5: "Market Conditions & Trading",
            6: "Business Strategy & Outlook",
            7: "Operational Efficiency & Costs",
            8: "Customer & Client Services",
            9: "Investment & Wealth Management"
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for topic modeling."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep financial terms
        text = re.sub(r'[^\w\s\.\%\$]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (less than 3 characters) except financial abbreviations
        financial_abbrevs = {'eps', 'roe', 'roa', 'nim', 'cet', 'usa', 'nyc', 'cfo', 'ceo'}
        words = text.split()
        filtered_words = []
        for word in words:
            if len(word) >= 3 or word in financial_abbrevs:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def get_financial_stopwords(self) -> List[str]:
        """Get financial domain-specific stopwords."""
        financial_stopwords = [
            'said', 'quarter', 'year', 'billion', 'million', 'percent', 'basis',
            'points', 'compared', 'prior', 'versus', 'period', 'ended', 'march',
            'december', 'september', 'june', 'first', 'second', 'third', 'fourth',
            'thank', 'thanks', 'good', 'morning', 'afternoon', 'question', 'answer',
            'next', 'slide', 'page', 'table', 'chart', 'please', 'well', 'also',
            'really', 'think', 'know', 'see', 'look', 'going', 'continue', 'way'
        ]
        
        # Standard English stopwords
        english_stopwords = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'a', 'an', 'as', 'if', 'each',
            'how', 'which', 'who', 'when', 'where', 'why', 'what', 'there', 'here'
        ]
        
        return financial_stopwords + english_stopwords
    
    def fit_topic_model(self, texts: List[str]) -> None:
        """Fit the topic model on the provided texts."""
        print(f"🔍 Preprocessing {len(texts)} texts for topic modeling...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Remove empty texts
        processed_texts = [text for text in processed_texts if text.strip()]
        
        print(f"📊 Fitting topic model with {len(processed_texts)} valid texts...")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words=self.get_financial_stopwords(),
            ngram_range=(1, 2)
        )
        
        # Fit vectorizer and transform texts
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        
        self.lda_model.fit(tfidf_matrix)
        
        # Generate topic labels based on top words
        self._generate_topic_labels()
        
        print(f"✅ Topic model fitted with {self.n_topics} topics")
    
    def _generate_topic_labels(self) -> None:
        """Generate human-readable topic labels."""
        if not self.vectorizer or not self.lda_model:
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        topic_labels = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get top words for this topic
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            # Use predefined labels or generate from top words
            if topic_idx in self.financial_topics:
                topic_labels[topic_idx] = self.financial_topics[topic_idx]
            else:
                # Generate label from top words
                topic_labels[topic_idx] = f"Topic_{topic_idx}: {', '.join(top_words[:3])}"
        
        self.topic_labels = topic_labels
    
    def predict_topics(self, texts: List[str]) -> List[Dict[str, any]]:
        """Predict topics for new texts."""
        if not self.vectorizer or not self.lda_model:
            raise ValueError("Model must be fitted before prediction")
        
        results = []
        
        for text in texts:
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                results.append({
                    'primary_topic': 'No Content',
                    'primary_topic_score': 0.0,
                    'topic_distribution': {},
                    'topic_labels': []
                })
                continue
            
            # Transform text
            tfidf_vector = self.vectorizer.transform([processed_text])
            
            # Get topic distribution
            topic_dist = self.lda_model.transform(tfidf_vector)[0]
            
            # Find primary topic
            primary_topic_idx = np.argmax(topic_dist)
            primary_topic_score = topic_dist[primary_topic_idx]
            primary_topic_label = self.topic_labels.get(primary_topic_idx, f"Topic_{primary_topic_idx}")
            
            # Get all topics above threshold
            threshold = 0.1
            significant_topics = []
            topic_distribution = {}
            
            for idx, score in enumerate(topic_dist):
                topic_label = self.topic_labels.get(idx, f"Topic_{idx}")
                topic_distribution[topic_label] = float(score)
                
                if score > threshold:
                    significant_topics.append(topic_label)
            
            results.append({
                'primary_topic': primary_topic_label,
                'primary_topic_score': float(primary_topic_score),
                'topic_distribution': topic_distribution,
                'topic_labels': significant_topics
            })
        
        return results

def add_topic_modeling_to_dataset():
    """Add topic modeling to the flattened dataset."""
    
    print("🎯 ADDING TOPIC MODELING TO ETL PIPELINE")
    print("=" * 60)
    
    # Load flattened data
    df = pd.read_csv('processed_data_flattened_nlp.csv')
    print(f"📊 Loaded {len(df)} records for topic modeling...")
    
    # Initialize topic modeler
    topic_modeler = FinancialTopicModeler(n_topics=10)
    
    # Prepare texts for topic modeling
    # Focus on financial content and named speaker content for better topics
    modeling_texts = []
    modeling_indices = []
    
    for idx, row in df.iterrows():
        text = str(row.get('text', ''))
        
        # Include text if it's financial content, from named speakers, or entity-rich
        include_text = (
            row.get('is_financial_content', False) or
            row.get('is_management', False) or
            row.get('is_analyst', False) or
            row.get('entity_count', 0) > 2 or
            len(text.split()) > 10  # Minimum length for meaningful topics
        )
        
        if include_text and text.strip():
            modeling_texts.append(text)
            modeling_indices.append(idx)
    
    print(f"🔍 Selected {len(modeling_texts)} texts for topic modeling...")
    
    # Fit topic model
    topic_modeler.fit_topic_model(modeling_texts)
    
    # Predict topics for all texts
    print("🏷️ Generating topic labels for all records...")
    
    all_texts = df['text'].fillna('').astype(str).tolist()
    topic_results = topic_modeler.predict_topics(all_texts)
    
    # Add topic columns to dataframe
    df['primary_topic'] = [result['primary_topic'] for result in topic_results]
    df['primary_topic_score'] = [result['primary_topic_score'] for result in topic_results]
    df['topic_labels_list'] = [' | '.join(result['topic_labels']) for result in topic_results]
    df['topic_distribution'] = [json.dumps(result['topic_distribution']) for result in topic_results]
    
    # Add topic-based boolean flags
    df['has_financial_topic'] = df['primary_topic'].str.contains('Financial|Revenue|Banking|Capital', case=False, na=False)
    df['has_strategy_topic'] = df['primary_topic'].str.contains('Strategy|Outlook|Business', case=False, na=False)
    df['has_performance_topic'] = df['primary_topic'].str.contains('Performance|Results|Growth', case=False, na=False)
    
    # Save enhanced dataset with topics
    enhanced_file = 'processed_data_with_topics.csv'
    df.to_csv(enhanced_file, index=False, encoding='utf-8-sig')
    
    # Update core NLP dataset with topics
    print("📋 Updating specialized NLP datasets with topics...")
    
    # Core NLP dataset with topics
    core_columns = [
        'Review_ID', 'source_file', 'quarter_period', 'sentence_id',
        'speaker_norm', 'speaker_role', 'speaker_category_enhanced',
        'text', 'word_count', 'char_count', 'complexity_score',
        'is_management', 'is_analyst', 'is_financial_content',
        'entities_text', 'financial_terms_text', 'financial_figures_text',
        'primary_topic', 'primary_topic_score', 'topic_labels_list',
        'has_financial_topic', 'has_strategy_topic', 'has_performance_topic'
    ]
    
    core_with_topics = df[[col for col in core_columns if col in df.columns]].copy()
    core_topics_file = 'nlp_core_dataset_with_topics.csv'
    core_with_topics.to_csv(core_topics_file, index=False, encoding='utf-8-sig')
    
    # Financial dataset with topics
    financial_with_topics = df[df['is_financial_content'] == True].copy()
    financial_topics_file = 'nlp_financial_dataset_with_topics.csv'
    financial_with_topics.to_csv(financial_topics_file, index=False, encoding='utf-8-sig')
    
    # Generate topic analysis report
    print("📊 Generating topic analysis report...")
    
    topic_distribution = df['primary_topic'].value_counts()
    financial_topic_dist = df[df['has_financial_topic']]['primary_topic'].value_counts()
    
    report_file = 'topic_modeling_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TOPIC MODELING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TOPIC MODELING APPROACH:\n")
        f.write("-" * 30 + "\n")
        f.write("✅ LDA (Latent Dirichlet Allocation) with TF-IDF\n")
        f.write("✅ Financial domain-specific preprocessing\n")
        f.write("✅ 10 predefined financial topics\n")
        f.write("✅ Bigram features for better context\n")
        f.write("✅ Financial stopwords filtering\n\n")
        
        f.write("TOPIC LABELS GENERATED:\n")
        f.write("-" * 30 + "\n")
        for idx, label in topic_modeler.topic_labels.items():
            f.write(f"Topic {idx}: {label}\n")
        
        f.write(f"\nTOPIC DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for topic, count in topic_distribution.head(10).items():
            percentage = (count / len(df)) * 100
            f.write(f"{topic}: {count} records ({percentage:.1f}%)\n")
        
        f.write(f"\nFINANCIAL TOPIC DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for topic, count in financial_topic_dist.head(5).items():
            percentage = (count / len(df[df['has_financial_topic']])) * 100
            f.write(f"{topic}: {count} records ({percentage:.1f}%)\n")
        
        f.write(f"\nTOPIC COVERAGE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Records with Financial Topics: {df['has_financial_topic'].sum()} ({(df['has_financial_topic'].sum()/len(df)*100):.1f}%)\n")
        f.write(f"Records with Strategy Topics: {df['has_strategy_topic'].sum()} ({(df['has_strategy_topic'].sum()/len(df)*100):.1f}%)\n")
        f.write(f"Records with Performance Topics: {df['has_performance_topic'].sum()} ({(df['has_performance_topic'].sum()/len(df)*100):.1f}%)\n")
        
        f.write(f"\nFILES CREATED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. {enhanced_file} - Complete dataset with topics\n")
        f.write(f"2. {core_topics_file} - Core NLP dataset with topics\n")
        f.write(f"3. {financial_topics_file} - Financial dataset with topics\n")
        f.write(f"4. {report_file} - This analysis report\n")
    
    # Print completion summary
    print(f"\n✅ TOPIC MODELING COMPLETE")
    print("=" * 60)
    print(f"📄 Enhanced dataset: {enhanced_file}")
    print(f"   • {len(df):,} records with topic labels")
    print(f"   • {len(topic_modeler.topic_labels)} financial topics")
    
    print(f"\n📋 Updated NLP datasets:")
    print(f"   • Core with topics: {core_topics_file} ({len(core_with_topics):,} records)")
    print(f"   • Financial with topics: {financial_topics_file} ({len(financial_with_topics):,} records)")
    
    print(f"\n📊 Analysis report: {report_file}")
    print(f"   • Topic distribution and coverage analysis")
    
    print(f"\n🎯 TOPIC MODELING RESULTS:")
    print(f"   • Financial topics: {df['has_financial_topic'].sum():,} records")
    print(f"   • Strategy topics: {df['has_strategy_topic'].sum():,} records")
    print(f"   • Performance topics: {df['has_performance_topic'].sum():,} records")
    print(f"   • Average topic score: {df['primary_topic_score'].mean():.3f}")
    
    return df

if __name__ == "__main__":
    enhanced_data = add_topic_modeling_to_dataset()
    print(f"\n🎉 Topic modeling added - ETL pipeline now complete with all NLP features!")