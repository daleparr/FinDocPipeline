"""
Advanced Risk Scoring Engine for Financial Risk Monitoring
Provides composite risk scores with multi-dimensional analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RiskScorer:
    """
    Advanced risk scoring engine for financial institutions
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Risk scoring configuration
        self.risk_weights = self.config.get('risk_scoring', {}).get('weights', {
            'sentiment_risk': 0.25,
            'topic_risk': 0.20,
            'speaker_risk': 0.15,
            'temporal_risk': 0.15,
            'anomaly_risk': 0.15,
            'volatility_risk': 0.10
        })
        
        # Risk thresholds
        self.risk_thresholds = self.config.get('risk_scoring', {}).get('thresholds', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        })
        
        # Speaker importance weights
        self.speaker_weights = self.config.get('speaker_weights', {
            'CEO': 1.0,
            'CFO': 0.9,
            'CRO': 0.8,
            'CCO': 0.7,
            'CTO': 0.6,
            'Analyst': 0.3,
            'UNKNOWN': 0.1
        })
        
        # Risk topic weights
        self.topic_risk_weights = self.config.get('risk_scoring', {}).get('topic_weights', {
            'credit_risk': 1.0,
            'operational_risk': 0.9,
            'market_risk': 0.8,
            'regulatory_risk': 0.8,
            'liquidity_risk': 0.9,
            'capital_management': 0.7,
            'strategic_risk': 0.6,
            'esg_sustainability': 0.4,
            'miscellaneous': 0.2
        })
        
        # Initialize scalers
        self.scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        
        logging.info("Risk scorer initialized with composite scoring methodology")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "risk_monitoring_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'risk_scoring': {
                'weights': {
                    'sentiment_risk': 0.25,
                    'topic_risk': 0.20,
                    'speaker_risk': 0.15,
                    'temporal_risk': 0.15,
                    'anomaly_risk': 0.15,
                    'volatility_risk': 0.10
                },
                'thresholds': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8
                },
                'topic_weights': {
                    'credit_risk': 1.0,
                    'operational_risk': 0.9,
                    'market_risk': 0.8,
                    'regulatory_risk': 0.8,
                    'liquidity_risk': 0.9,
                    'capital_management': 0.7,
                    'strategic_risk': 0.6,
                    'esg_sustainability': 0.4,
                    'miscellaneous': 0.2
                }
            },
            'speaker_weights': {
                'CEO': 1.0,
                'CFO': 0.9,
                'CRO': 0.8,
                'CCO': 0.7,
                'CTO': 0.6,
                'Analyst': 0.3,
                'UNKNOWN': 0.1
            }
        }
    
    def calculate_sentiment_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate sentiment-based risk scores
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Series with sentiment risk scores
        """
        sentiment_risk = pd.Series(0.0, index=df.index)
        
        # Negative sentiment contribution
        if 'sentiment_negative' in df.columns:
            sentiment_risk += df['sentiment_negative'] * 0.4
        
        # Risk escalation contribution
        if 'risk_escalation_score' in df.columns:
            # Normalize and scale risk escalation
            risk_escalation_norm = np.clip(df['risk_escalation_score'], 0, 1)
            sentiment_risk += risk_escalation_norm * 0.3
        
        # Stress indicators
        if 'stress_score' in df.columns:
            stress_norm = np.clip(df['stress_score'], 0, 1)
            sentiment_risk += stress_norm * 0.2
        
        # Uncertainty and hedging
        if 'uncertainty_score' in df.columns:
            uncertainty_norm = np.clip(df['uncertainty_score'], 0, 1)
            sentiment_risk += uncertainty_norm * 0.1
        
        # Inverse confidence (low confidence = higher risk)
        if 'confidence_score' in df.columns:
            confidence_risk = 1 - np.clip(df['confidence_score'], 0, 1)
            sentiment_risk += confidence_risk * 0.1
        
        # Normalize to [0, 1] range
        sentiment_risk = np.clip(sentiment_risk, 0, 1)
        
        return sentiment_risk
    
    def calculate_topic_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate topic-based risk scores
        
        Args:
            df: DataFrame with topic modeling results
            
        Returns:
            Series with topic risk scores
        """
        topic_risk = pd.Series(0.0, index=df.index)
        
        if 'final_topic' not in df.columns:
            return topic_risk
        
        # Map topics to risk weights
        for idx, topic in df['final_topic'].items():
            topic_weight = self.topic_risk_weights.get(str(topic).lower(), 0.2)
            
            # Base risk from topic type
            base_risk = topic_weight * 0.6
            
            # Confidence adjustment
            if 'topic_confidence' in df.columns:
                confidence = df.loc[idx, 'topic_confidence']
                if not pd.isna(confidence):
                    # Higher confidence in high-risk topics increases risk
                    confidence_adjustment = confidence * 0.4 if topic_weight > 0.7 else confidence * 0.2
                    base_risk += confidence_adjustment
            
            topic_risk.loc[idx] = min(base_risk, 1.0)
        
        return topic_risk
    
    def calculate_speaker_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate speaker-weighted risk scores
        
        Args:
            df: DataFrame with speaker information
            
        Returns:
            Series with speaker risk scores
        """
        speaker_risk = pd.Series(0.0, index=df.index)
        
        if 'speaker_norm' not in df.columns:
            return speaker_risk
        
        # Calculate base sentiment risk for weighting
        base_sentiment_risk = self.calculate_sentiment_risk(df)
        
        # Apply speaker weights
        for idx, speaker in df['speaker_norm'].items():
            speaker_weight = self.speaker_weights.get(str(speaker), 0.1)
            
            # Higher weight speakers contribute more to risk when expressing negative sentiment
            weighted_risk = base_sentiment_risk.loc[idx] * speaker_weight
            speaker_risk.loc[idx] = weighted_risk
        
        return speaker_risk
    
    def calculate_temporal_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate temporal risk patterns
        
        Args:
            df: DataFrame with temporal information
            
        Returns:
            Series with temporal risk scores
        """
        temporal_risk = pd.Series(0.0, index=df.index)
        
        # If we have quarter information, analyze temporal patterns
        if 'quarter' in df.columns and 'bank' in df.columns:
            # Group by institution and quarter
            for institution in df['bank'].unique():
                inst_data = df[df['bank'] == institution]
                
                if len(inst_data) < 2:
                    continue
                
                # Calculate quarter-over-quarter changes in risk metrics
                for quarter in inst_data['quarter'].unique():
                    quarter_data = inst_data[inst_data['quarter'] == quarter]
                    
                    # Calculate average negative sentiment for this quarter
                    if 'sentiment_negative' in df.columns:
                        quarter_neg_sentiment = quarter_data['sentiment_negative'].mean()
                        
                        # Compare with institution's historical average
                        hist_avg = inst_data['sentiment_negative'].mean()
                        
                        if quarter_neg_sentiment > hist_avg * 1.2:  # 20% above average
                            temporal_risk.loc[quarter_data.index] += 0.3
                    
                    # Check for risk escalation trends
                    if 'risk_escalation_score' in df.columns:
                        quarter_risk = quarter_data['risk_escalation_score'].mean()
                        hist_risk_avg = inst_data['risk_escalation_score'].mean()
                        
                        if quarter_risk > hist_risk_avg * 1.5:  # 50% above average
                            temporal_risk.loc[quarter_data.index] += 0.4
        
        # Recent bias - more recent quarters get higher weight if risky
        if 'quarter' in df.columns:
            # Simple recency weighting based on quarter
            quarters = df['quarter'].unique()
            if len(quarters) > 1:
                sorted_quarters = sorted(quarters)
                for i, quarter in enumerate(sorted_quarters):
                    recency_weight = (i + 1) / len(sorted_quarters) * 0.3
                    quarter_mask = df['quarter'] == quarter
                    temporal_risk.loc[quarter_mask] += recency_weight
        
        temporal_risk = np.clip(temporal_risk, 0, 1)
        return temporal_risk
    
    def calculate_anomaly_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate anomaly-based risk scores
        
        Args:
            df: DataFrame with anomaly detection results
            
        Returns:
            Series with anomaly risk scores
        """
        anomaly_risk = pd.Series(0.0, index=df.index)
        
        # If anomaly scores are available, use them directly
        if 'anomaly_score' in df.columns:
            anomaly_risk = df['anomaly_score'].fillna(0)
        elif 'is_anomaly' in df.columns:
            # Binary anomaly indicator
            anomaly_risk = df['is_anomaly'].fillna(0).astype(float)
        else:
            # Calculate simple statistical anomalies
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            risk_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                        for keyword in ['risk', 'negative', 'stress', 'uncertainty'])]
            
            if risk_cols:
                # Calculate z-scores for risk-related columns
                for col in risk_cols:
                    z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                    # Anomaly if z-score > 2
                    col_anomaly = (z_scores > 2).astype(float) * 0.5
                    anomaly_risk += col_anomaly
                
                anomaly_risk = np.clip(anomaly_risk, 0, 1)
        
        return anomaly_risk
    
    def calculate_volatility_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility-based risk scores
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            Series with volatility risk scores
        """
        volatility_risk = pd.Series(0.0, index=df.index)
        
        if 'bank' not in df.columns:
            return volatility_risk
        
        # Calculate volatility for each institution
        for institution in df['bank'].unique():
            inst_data = df[df['bank'] == institution]
            
            if len(inst_data) < 3:  # Need at least 3 points for volatility
                continue
            
            # Calculate volatility in key metrics
            volatility_metrics = ['sentiment_negative', 'risk_escalation_score', 'stress_score']
            total_volatility = 0
            valid_metrics = 0
            
            for metric in volatility_metrics:
                if metric in inst_data.columns:
                    metric_values = inst_data[metric].dropna()
                    if len(metric_values) >= 3:
                        # Calculate coefficient of variation (std/mean)
                        cv = metric_values.std() / (metric_values.mean() + 0.001)
                        total_volatility += cv
                        valid_metrics += 1
            
            if valid_metrics > 0:
                avg_volatility = total_volatility / valid_metrics
                # Normalize volatility to [0, 1] range
                normalized_volatility = min(avg_volatility / 2.0, 1.0)  # Assume CV > 2 is very high
                volatility_risk.loc[inst_data.index] = normalized_volatility
        
        return volatility_risk
    
    def calculate_composite_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite risk scores using all risk dimensions
        
        Args:
            df: DataFrame with all analysis results
            
        Returns:
            DataFrame with risk scores and components
        """
        results_df = df.copy()
        
        # Calculate individual risk components
        sentiment_risk = self.calculate_sentiment_risk(df)
        topic_risk = self.calculate_topic_risk(df)
        speaker_risk = self.calculate_speaker_risk(df)
        temporal_risk = self.calculate_temporal_risk(df)
        anomaly_risk = self.calculate_anomaly_risk(df)
        volatility_risk = self.calculate_volatility_risk(df)
        
        # Add individual components to results
        results_df['risk_sentiment'] = sentiment_risk
        results_df['risk_topic'] = topic_risk
        results_df['risk_speaker'] = speaker_risk
        results_df['risk_temporal'] = temporal_risk
        results_df['risk_anomaly'] = anomaly_risk
        results_df['risk_volatility'] = volatility_risk
        
        # Calculate weighted composite score
        composite_score = (
            sentiment_risk * self.risk_weights['sentiment_risk'] +
            topic_risk * self.risk_weights['topic_risk'] +
            speaker_risk * self.risk_weights['speaker_risk'] +
            temporal_risk * self.risk_weights['temporal_risk'] +
            anomaly_risk * self.risk_weights['anomaly_risk'] +
            volatility_risk * self.risk_weights['volatility_risk']
        )
        
        results_df['risk_score'] = composite_score
        
        # Categorize risk levels
        risk_categories = []
        for score in composite_score:
            if score <= self.risk_thresholds['low']:
                risk_categories.append('low')
            elif score <= self.risk_thresholds['medium']:
                risk_categories.append('medium')
            elif score <= self.risk_thresholds['high']:
                risk_categories.append('high')
            else:
                risk_categories.append('critical')
        
        results_df['risk_category'] = risk_categories
        
        # Calculate risk percentile within dataset
        results_df['risk_percentile'] = results_df['risk_score'].rank(pct=True)
        
        return results_df
    
    def generate_institution_risk_profile(self, df: pd.DataFrame, institution: str) -> Dict[str, Any]:
        """
        Generate comprehensive risk profile for an institution
        
        Args:
            df: DataFrame with risk scores
            institution: Institution name
            
        Returns:
            Dictionary with institution risk profile
        """
        inst_data = df[df['bank'] == institution] if 'bank' in df.columns else df
        
        if len(inst_data) == 0:
            return {}
        
        # Overall statistics
        profile = {
            'institution': institution,
            'analysis_date': datetime.now().isoformat(),
            'total_records': len(inst_data),
            'overall_risk': {
                'mean_score': float(inst_data['risk_score'].mean()),
                'max_score': float(inst_data['risk_score'].max()),
                'min_score': float(inst_data['risk_score'].min()),
                'std_score': float(inst_data['risk_score'].std()),
                'current_score': float(inst_data['risk_score'].iloc[-1]) if len(inst_data) > 0 else 0
            },
            'risk_distribution': {},
            'component_analysis': {},
            'temporal_trends': {},
            'high_risk_indicators': [],
            'recommendations': []
        }
        
        # Risk category distribution
        risk_dist = inst_data['risk_category'].value_counts()
        profile['risk_distribution'] = risk_dist.to_dict()
        
        # Component analysis
        risk_components = ['risk_sentiment', 'risk_topic', 'risk_speaker', 
                          'risk_temporal', 'risk_anomaly', 'risk_volatility']
        
        for component in risk_components:
            if component in inst_data.columns:
                profile['component_analysis'][component] = {
                    'mean': float(inst_data[component].mean()),
                    'max': float(inst_data[component].max()),
                    'contribution': float(inst_data[component].mean() * 
                                        self.risk_weights.get(component.replace('risk_', '') + '_risk', 0))
                }
        
        # Temporal trends (if quarter data available)
        if 'quarter' in inst_data.columns:
            quarterly_risk = inst_data.groupby('quarter')['risk_score'].agg(['mean', 'max', 'count'])
            profile['temporal_trends'] = quarterly_risk.to_dict('index')
        
        # High risk indicators
        high_risk_records = inst_data[inst_data['risk_score'] > self.risk_thresholds['high']]
        if len(high_risk_records) > 0:
            profile['high_risk_indicators'] = [
                {
                    'quarter': str(row.get('quarter', 'Unknown')),
                    'risk_score': float(row['risk_score']),
                    'risk_category': str(row['risk_category']),
                    'primary_driver': self._identify_primary_risk_driver(row),
                    'text_preview': str(row.get('text', ''))[:100] + '...' if 'text' in row and len(str(row['text'])) > 100 else str(row.get('text', ''))
                }
                for _, row in high_risk_records.head(10).iterrows()
            ]
        
        # Generate recommendations
        profile['recommendations'] = self._generate_recommendations(profile)
        
        return profile
    
    def _identify_primary_risk_driver(self, row: pd.Series) -> str:
        """Identify the primary risk driver for a record"""
        risk_components = {
            'sentiment': row.get('risk_sentiment', 0),
            'topic': row.get('risk_topic', 0),
            'speaker': row.get('risk_speaker', 0),
            'temporal': row.get('risk_temporal', 0),
            'anomaly': row.get('risk_anomaly', 0),
            'volatility': row.get('risk_volatility', 0)
        }
        
        # Find component with highest contribution
        max_component = max(risk_components.items(), key=lambda x: x[1])
        return max_component[0]
    
    def _generate_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk profile"""
        recommendations = []
        
        overall_risk = profile['overall_risk']['mean_score']
        
        if overall_risk > self.risk_thresholds['high']:
            recommendations.append("Immediate risk assessment required - overall risk score is high")
            recommendations.append("Increase monitoring frequency to weekly")
            recommendations.append("Consider regulatory notification")
        elif overall_risk > self.risk_thresholds['medium']:
            recommendations.append("Enhanced monitoring recommended")
            recommendations.append("Review risk management procedures")
        
        # Component-specific recommendations
        components = profile.get('component_analysis', {})
        
        if components.get('risk_sentiment', {}).get('mean', 0) > 0.7:
            recommendations.append("High negative sentiment detected - review communication strategy")
        
        if components.get('risk_topic', {}).get('mean', 0) > 0.7:
            recommendations.append("High-risk topics prevalent - focus on risk mitigation")
        
        if components.get('risk_anomaly', {}).get('mean', 0) > 0.5:
            recommendations.append("Anomalous patterns detected - investigate underlying causes")
        
        if components.get('risk_volatility', {}).get('mean', 0) > 0.6:
            recommendations.append("High volatility in risk metrics - stabilize risk management")
        
        # Temporal recommendations
        temporal_trends = profile.get('temporal_trends', {})
        if temporal_trends:
            recent_quarters = sorted(temporal_trends.keys())[-2:]
            if len(recent_quarters) >= 2:
                recent_risk = temporal_trends[recent_quarters[-1]]['mean']
                previous_risk = temporal_trends[recent_quarters[-2]]['mean']
                
                if recent_risk > previous_risk * 1.2:
                    recommendations.append("Risk trend is increasing - implement corrective measures")
        
        return recommendations
    
    def generate_comparative_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comparative risk analysis across institutions
        
        Args:
            df: DataFrame with risk scores for multiple institutions
            
        Returns:
            Dictionary with comparative analysis
        """
        if 'bank' not in df.columns:
            return {}
        
        analysis = {
            'analysis_date': datetime.now().isoformat(),
            'institution_rankings': {},
            'risk_distribution': {},
            'peer_comparison': {},
            'industry_benchmarks': {}
        }
        
        # Institution rankings
        inst_risk = df.groupby('bank')['risk_score'].agg(['mean', 'max', 'std', 'count'])
        inst_risk_sorted = inst_risk.sort_values('mean', ascending=False)
        
        analysis['institution_rankings'] = {
            'by_mean_risk': inst_risk_sorted['mean'].to_dict(),
            'by_max_risk': inst_risk.sort_values('max', ascending=False)['max'].to_dict(),
            'by_volatility': inst_risk.sort_values('std', ascending=False)['std'].to_dict()
        }
        
        # Overall risk distribution
        risk_dist = df['risk_category'].value_counts()
        analysis['risk_distribution'] = risk_dist.to_dict()
        
        # Industry benchmarks
        analysis['industry_benchmarks'] = {
            'mean_risk_score': float(df['risk_score'].mean()),
            'median_risk_score': float(df['risk_score'].median()),
            'risk_score_std': float(df['risk_score'].std()),
            'high_risk_rate': float((df['risk_score'] > self.risk_thresholds['high']).mean()),
            'critical_risk_rate': float((df['risk_score'] > self.risk_thresholds['high'] * 1.25).mean())
        }
        
        return analysis

def get_risk_scorer(config_path: Optional[str] = None) -> RiskScorer:
    """Get risk scorer instance"""
    return RiskScorer(config_path)