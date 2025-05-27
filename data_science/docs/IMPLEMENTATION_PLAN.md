# Risk Monitoring Pipeline: Implementation Plan

## Overview

This document provides a detailed implementation plan for the risk monitoring pipeline, mapping the architecture to specific code modules, configuration files, and development tasks.

## 1. Project Structure

```
data_science/
├── config/
│   ├── risk_monitoring_config.yaml
│   ├── seed_topics.yaml
│   ├── bank_metadata.yaml
│   └── model_configs/
├── scripts/
│   ├── topic_modeling/
│   │   ├── seed_topic_classifier.py
│   │   ├── emergent_topic_discovery.py
│   │   ├── topic_coherence_validator.py
│   │   └── hybrid_topic_engine.py
│   ├── sentiment_analysis/
│   │   ├── finbert_analyzer.py
│   │   ├── tone_analyzer.py
│   │   ├── hedging_detector.py
│   │   └── sentiment_aggregator.py
│   ├── anomaly_detection/
│   │   ├── statistical_anomaly_detector.py
│   │   ├── change_point_detector.py
│   │   ├── cross_bank_correlator.py
│   │   └── risk_scorer.py
│   ├── pipeline/
│   │   ├── etl_orchestrator.py
│   │   ├── nlp_processor.py
│   │   ├── alert_generator.py
│   │   └── dashboard_builder.py
│   └── utils/
│       ├── statistical_utils.py
│       ├── text_preprocessing.py
│       └── validation_utils.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_topic_modeling_analysis.ipynb
│   ├── 03_sentiment_analysis_validation.ipynb
│   ├── 04_anomaly_detection_tuning.ipynb
│   └── 05_end_to_end_pipeline_testing.ipynb
├── models/
│   ├── topic_models/
│   ├── sentiment_models/
│   ├── anomaly_detectors/
│   └── model_registry.json
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── sentences/
│   │   ├── topics/
│   │   ├── sentiment/
│   │   └── alerts/
│   └── external/
│       ├── financial_lexicons/
│       └── regulatory_calendars/
└── tests/
    ├── test_topic_modeling.py
    ├── test_sentiment_analysis.py
    ├── test_anomaly_detection.py
    └── test_integration.py
```

## 2. Configuration Management

### 2.1 Main Configuration File

```yaml
# data_science/config/risk_monitoring_config.yaml
system:
  name: "Financial Risk Monitoring Pipeline"
  version: "1.0.0"
  environment: "production"

data_sources:
  banks:
    us_banks:
      - jpmorgan_chase
      - bank_of_america
      - wells_fargo
      - citigroup
    european_banks:
      - hsbc
      - barclays
      - deutsche_bank
      - ubs
  
  quarters:
    - Q1_2023
    - Q2_2023
    - Q3_2023
    - Q4_2023
    - Q1_2024
    - Q2_2024
    - Q3_2024
    - Q4_2024
    - Q1_2025

processing:
  batch_size: 1000
  parallel_workers: 4
  memory_limit: "16GB"
  
topic_modeling:
  seed_threshold: 3  # minimum keyword matches
  emergent_min_cluster_size: 50
  coherence_threshold: 0.4
  max_topics: 50
  
sentiment_analysis:
  model_name: "ProsusAI/finbert"
  batch_size: 32
  confidence_threshold: 0.7
  
anomaly_detection:
  contamination_rate: 0.05
  min_historical_quarters: 8
  significance_level: 0.01
  effect_size_threshold: 0.3
  
alerts:
  severity_levels:
    critical: 0.8
    high: 0.6
    medium: 0.4
  cross_bank_threshold: 2
  persistence_quarters: 2

output:
  save_intermediate: true
  export_formats: ["parquet", "json"]
  dashboard_refresh_hours: 6
```

### 2.2 Seed Topics Configuration

```yaml
# data_science/config/seed_topics.yaml
seed_topics:
  credit_risk:
    keywords:
      primary: [loan, credit, default, provision, npl, charge-off, delinquency, impairment]
      secondary: [borrower, underwriting, recovery, workout, restructuring]
    weight: 1.0
    min_confidence: 0.7
    
  operational_risk:
    keywords:
      primary: [cyber, fraud, compliance, regulatory, operational, control, breach]
      secondary: [incident, failure, disruption, vulnerability, governance]
    weight: 1.0
    min_confidence: 0.7
    
  market_risk:
    keywords:
      primary: [trading, market, volatility, var, stress, liquidity, correlation]
      secondary: [portfolio, exposure, hedging, derivative, counterparty]
    weight: 1.0
    min_confidence: 0.7
    
  regulatory_risk:
    keywords:
      primary: [regulation, capital, basel, stress test, ccar, cecl, ifrs]
      secondary: [supervisor, examination, enforcement, penalty, consent]
    weight: 1.0
    min_confidence: 0.7
    
  macroeconomic_risk:
    keywords:
      primary: [inflation, recession, gdp, unemployment, interest rate, fed]
      secondary: [monetary policy, fiscal, economic, downturn, cycle]
    weight: 1.0
    min_confidence: 0.6
    
  climate_risk:
    keywords:
      primary: [climate, esg, sustainability, carbon, environmental, transition]
      secondary: [physical risk, stranded assets, green finance, tcfd]
    weight: 0.8
    min_confidence: 0.6
    
  digital_transformation:
    keywords:
      primary: [digital, technology, fintech, blockchain, ai, automation, cloud]
      secondary: [innovation, disruption, platform, api, data analytics]
    weight: 0.8
    min_confidence: 0.6
    
  geopolitical_risk:
    keywords:
      primary: [geopolitical, sanctions, trade war, brexit, china, russia]
      secondary: [political, sovereign, country risk, emerging markets]
    weight: 0.7
    min_confidence: 0.6
```

## 3. Core Implementation Modules

### 3.1 Topic Modeling Engine

```python
# data_science/scripts/topic_modeling/hybrid_topic_engine.py
"""
Hybrid Topic Modeling Engine combining seed-based and emergent discovery
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from typing import Dict, List, Tuple, Optional

class HybridTopicEngine:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.seed_topics = self._load_seed_topics()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bertopic_model = None
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_seed_topics(self) -> Dict:
        with open('config/seed_topics.yaml', 'r') as f:
            return yaml.safe_load(f)['seed_topics']
    
    def assign_seed_topics(self, sentences: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assign seed topics based on keyword matching
        Returns: (seed_assigned_df, remaining_df)
        """
        seed_assigned = []
        remaining = []
        
        for idx, row in sentences.iterrows():
            text = row['text'].lower()
            best_topic = None
            best_score = 0
            
            for topic_name, topic_config in self.seed_topics.items():
                score = self._calculate_seed_score(text, topic_config)
                if score >= topic_config['min_confidence'] and score > best_score:
                    best_topic = topic_name
                    best_score = score
            
            if best_topic:
                row_copy = row.copy()
                row_copy['topic_seed'] = best_topic
                row_copy['topic_confidence'] = best_score
                seed_assigned.append(row_copy)
            else:
                remaining.append(row)
        
        return pd.DataFrame(seed_assigned), pd.DataFrame(remaining)
    
    def _calculate_seed_score(self, text: str, topic_config: Dict) -> float:
        """Calculate seed topic matching score"""
        primary_matches = sum(1 for kw in topic_config['keywords']['primary'] if kw in text)
        secondary_matches = sum(1 for kw in topic_config['keywords']['secondary'] if kw in text)
        
        # Weighted score
        score = (primary_matches * 1.0 + secondary_matches * 0.5) / len(topic_config['keywords']['primary'])
        return min(score, 1.0)
    
    def discover_emergent_topics(self, sentences: pd.DataFrame) -> pd.DataFrame:
        """
        Use BERTopic to discover emergent topics in remaining sentences
        """
        if len(sentences) < self.config['topic_modeling']['emergent_min_cluster_size']:
            # Not enough data for clustering
            sentences['topic_emergent'] = 'misc'
            sentences['topic_confidence'] = 0.0
            return sentences
        
        # Prepare texts for BERTopic
        texts = sentences['text'].tolist()
        
        # Configure BERTopic
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=1000
        )
        
        self.bertopic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=self.config['topic_modeling']['emergent_min_cluster_size'],
            nr_topics=self.config['topic_modeling']['max_topics'],
            calculate_probabilities=True
        )
        
        # Fit and transform
        topics, probabilities = self.bertopic_model.fit_transform(texts)
        
        # Assign results
        sentences = sentences.copy()
        sentences['topic_emergent'] = [f"emergent_{topic}" if topic != -1 else "misc" for topic in topics]
        sentences['topic_confidence'] = [max(prob) if prob is not None else 0.0 for prob in probabilities]
        
        return sentences
    
    def validate_topic_coherence(self) -> Dict[str, float]:
        """
        Calculate coherence scores for discovered topics
        """
        if self.bertopic_model is None:
            return {}
        
        # Get topic coherence (BERTopic built-in)
        coherence_scores = {}
        try:
            topics = self.bertopic_model.get_topics()
            for topic_id in topics.keys():
                if topic_id != -1:  # Skip outlier topic
                    # Calculate c_v coherence for this topic
                    coherence = self.bertopic_model.get_topic_info()
                    coherence_scores[f"emergent_{topic_id}"] = 0.5  # Placeholder
        except Exception as e:
            print(f"Error calculating coherence: {e}")
        
        return coherence_scores
    
    def process_quarter_data(self, bank: str, quarter: str, sentences: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing function for a quarter's data
        """
        print(f"Processing {bank} {quarter}: {len(sentences)} sentences")
        
        # Step 1: Assign seed topics
        seed_assigned, remaining = self.assign_seed_topics(sentences)
        print(f"Seed topics assigned: {len(seed_assigned)} sentences")
        
        # Step 2: Discover emergent topics in remaining sentences
        if len(remaining) > 0:
            emergent_assigned = self.discover_emergent_topics(remaining)
            print(f"Emergent topics discovered: {len(emergent_assigned)} sentences")
            
            # Combine results
            result = pd.concat([seed_assigned, emergent_assigned], ignore_index=True)
        else:
            result = seed_assigned
        
        # Step 3: Validate coherence
        coherence_scores = self.validate_topic_coherence()
        
        # Add metadata
        result['bank'] = bank
        result['quarter'] = quarter
        result['processing_timestamp'] = pd.Timestamp.now()
        
        return result, coherence_scores
```

### 3.2 Sentiment Analysis Engine

```python
# data_science/scripts/sentiment_analysis/finbert_analyzer.py
"""
FinBERT-based sentiment analysis for financial texts
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from typing import Dict, List, Tuple
import logging

class FinBERTAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def analyze_sentiment(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment for a list of texts
        Returns list of {label, score, confidence} dicts
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.sentiment_pipeline(batch)
            
            for result in batch_results:
                # FinBERT returns positive, negative, neutral
                processed_result = {
                    'sentiment_label': result['label'].lower(),
                    'sentiment_score': result['score'],
                    'sentiment_confidence': result['score']
                }
                results.append(processed_result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Add sentiment analysis to a DataFrame
        """
        texts = df[text_column].fillna('').tolist()
        sentiment_results = self.analyze_sentiment(texts)
        
        # Add results to DataFrame
        df_copy = df.copy()
        for i, result in enumerate(sentiment_results):
            for key, value in result.items():
                df_copy.loc[i, key] = value
        
        return df_copy
```

### 3.3 Anomaly Detection Engine

```python
# data_science/scripts/anomaly_detection/statistical_anomaly_detector.py
"""
Statistical anomaly detection for risk signals
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
from typing import Dict, List, Tuple, Optional
import warnings

class StatisticalAnomalyDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.contamination_rate = config['anomaly_detection']['contamination_rate']
        self.significance_level = config['anomaly_detection']['significance_level']
        self.min_quarters = config['anomaly_detection']['min_historical_quarters']
        
    def detect_prevalence_anomalies(self, topic_trends: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in topic prevalence using Isolation Forest
        """
        anomalies = []
        
        for topic in topic_trends['topic'].unique():
            topic_data = topic_trends[topic_trends['topic'] == topic].copy()
            
            if len(topic_data) < self.min_quarters:
                continue
            
            # Prepare features for anomaly detection
            features = ['prevalence', 'quarter_over_quarter_change', 'sentiment_score']
            X = topic_data[features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.contamination_rate,
                random_state=42
            )
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            anomaly_scores = iso_forest.score_samples(X_scaled)
            
            # Mark anomalies
            topic_data['is_anomaly'] = anomaly_labels == -1
            topic_data['anomaly_score'] = anomaly_scores
            
            anomalies.append(topic_data)
        
        return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()
    
    def detect_sentiment_regime_changes(self, sentiment_trends: pd.DataFrame) -> Dict[str, List]:
        """
        Detect regime changes in sentiment using change point detection
        """
        change_points = {}
        
        for topic in sentiment_trends['topic'].unique():
            topic_data = sentiment_trends[sentiment_trends['topic'] == topic].copy()
            topic_data = topic_data.sort_values('quarter')
            
            if len(topic_data) < self.min_quarters:
                continue
            
            # Prepare time series
            sentiment_series = topic_data['sentiment_score'].fillna(0).values
            
            # Change point detection using PELT algorithm
            algo = rpt.Pelt(model="rbf").fit(sentiment_series)
            change_points_idx = algo.predict(pen=10)
            
            # Convert to quarters
            quarters = topic_data['quarter'].tolist()
            change_point_quarters = [quarters[idx-1] for idx in change_points_idx[:-1]]
            
            change_points[topic] = change_point_quarters
        
        return change_points
    
    def calculate_cross_bank_correlation(self, bank_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate cross-bank correlations for risk signals
        """
        correlations = []
        
        # Get all unique topics across banks
        all_topics = set()
        for bank_df in bank_data.values():
            all_topics.update(bank_df['topic'].unique())
        
        for topic in all_topics:
            topic_correlations = {}
            bank_series = {}
            
            # Extract time series for each bank
            for bank, df in bank_data.items():
                topic_data = df[df['topic'] == topic].sort_values('quarter')
                if len(topic_data) >= 4:  # Minimum quarters for correlation
                    bank_series[bank] = topic_data.set_index('quarter')['prevalence']
            
            if len(bank_series) < 2:
                continue
            
            # Calculate pairwise correlations
            banks = list(bank_series.keys())
            for i in range(len(banks)):
                for j in range(i+1, len(banks)):
                    bank1, bank2 = banks[i], banks[j]
                    
                    # Align time series
                    common_quarters = bank_series[bank1].index.intersection(bank_series[bank2].index)
                    if len(common_quarters) >= 4:
                        series1 = bank_series[bank1].loc[common_quarters]
                        series2 = bank_series[bank2].loc[common_quarters]
                        
                        correlation, p_value = stats.pearsonr(series1, series2)
                        
                        correlations.append({
                            'topic': topic,
                            'bank1': bank1,
                            'bank2': bank2,
                            'correlation': correlation,
                            'p_value': p_value,
                            'is_significant': p_value < self.significance_level,
                            'quarters_count': len(common_quarters)
                        })
        
        return pd.DataFrame(correlations)
    
    def calculate_risk_scores(self, anomalies: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite risk scores
        """
        risk_scores = []
        
        for _, row in anomalies.iterrows():
            if not row['is_anomaly']:
                continue
            
            topic = row['topic']
            bank = row['bank']
            quarter = row['quarter']
            
            # Base anomaly score
            base_score = abs(row['anomaly_score'])
            
            # Cross-bank correlation boost
            topic_correlations = correlations[correlations['topic'] == topic]
            bank_correlations = topic_correlations[
                (topic_correlations['bank1'] == bank) | (topic_correlations['bank2'] == bank)
            ]
            
            correlation_boost = 0
            if len(bank_correlations) > 0:
                significant_correlations = bank_correlations[bank_correlations['is_significant']]
                correlation_boost = len(significant_correlations) * 0.1
            
            # Sentiment deterioration penalty
            sentiment_penalty = 0
            if 'sentiment_score' in row and row['sentiment_score'] < -0.5:
                sentiment_penalty = 0.2
            
            # Calculate final risk score
            risk_score = min(base_score + correlation_boost + sentiment_penalty, 1.0)
            
            risk_scores.append({
                'bank': bank,
                'quarter': quarter,
                'topic': topic,
                'risk_score': risk_score,
                'base_anomaly_score': base_score,
                'correlation_boost': correlation_boost,
                'sentiment_penalty': sentiment_penalty,
                'cross_bank_count': len(bank_correlations)
            })
        
        return pd.DataFrame(risk_scores)
```

## 4. Implementation Timeline

### Week 1-2: Foundation Setup
- [ ] Set up project structure and configuration files
- [ ] Implement basic data loading and preprocessing utilities
- [ ] Create seed topic classification module
- [ ] Set up testing framework

### Week 3-4: Core NLP Components
- [ ] Implement FinBERT sentiment analysis
- [ ] Build tone analysis and hedging detection
- [ ] Integrate BERTopic for emergent topic discovery
- [ ] Create topic coherence validation

### Week 5-6: Statistical Analysis
- [ ] Implement anomaly detection algorithms
- [ ] Build change point detection for sentiment
- [ ] Create cross-bank correlation analysis
- [ ] Develop risk scoring framework

### Week 7-8: Pipeline Integration
- [ ] Build end-to-end processing pipeline
- [ ] Implement alert generation system
- [ ] Create dashboard data preparation
- [ ] Add model performance monitoring

### Week 9-10: Validation & Testing
- [ ] Comprehensive testing of all components
- [ ] Historical backtesting on crisis periods
- [ ] Performance optimization
- [ ] Documentation completion

### Week 11-12: Production Deployment
- [ ] Production environment setup
- [ ] Monitoring and alerting configuration
- [ ] User training and documentation
- [ ] Go-live and initial monitoring

## 5. Success Criteria

### Technical Metrics
- **Topic Coherence**: Average c_v score > 0.5 for emergent topics
- **Sentiment Accuracy**: F1 score > 0.85 on financial sentiment validation set
- **Processing Speed**: Complete quarterly analysis in < 2 hours
- **Alert Precision**: True positive rate > 70% for high-severity alerts

### Business Metrics
- **Early Detection**: Identify risk signals 1-2 quarters before market recognition
- **Coverage**: Successfully monitor 95% of relevant risk themes
- **Noise Reduction**: False positive rate < 20% for critical alerts
- **Actionability**: 80% of high-severity alerts lead to investigation

## 6. Risk Mitigation Strategies

### Technical Risks
- **Model Performance Degradation**: Implement automated model monitoring and retraining
- **Data Quality Issues**: Build comprehensive data validation and quality checks
- **Scalability Bottlenecks**: Design for horizontal scaling from the start

### Business Risks
- **False Alert Fatigue**: Implement progressive alert severity and explanation features
- **Regulatory Compliance**: Ensure model explainability and audit trails
- **Bias and Fairness**: Regular bias testing across banks and regions

---

*This implementation plan provides a roadmap for building a production-ready risk monitoring system with clear milestones, success criteria, and risk mitigation strategies.*