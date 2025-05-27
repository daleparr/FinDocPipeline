"""
Advanced Time Series Analysis Engine for Financial Risk Monitoring
Provides trend detection, seasonality analysis, and forecasting capabilities
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """
    Advanced time series analyzer for financial risk monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Time series configuration
        self.trend_window = self.config.get('time_series', {}).get('trend_window', 4)
        self.seasonality_periods = self.config.get('time_series', {}).get('seasonality_periods', [4, 12])
        self.anomaly_threshold = self.config.get('time_series', {}).get('anomaly_threshold', 2.5)
        self.min_periods = self.config.get('time_series', {}).get('min_periods', 3)
        
        # Statistical thresholds
        self.significance_level = self.config.get('statistics', {}).get('significance_level', 0.05)
        self.correlation_threshold = self.config.get('statistics', {}).get('correlation_threshold', 0.3)
        
        # Initialize scalers and models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        
        logging.info("Time series analyzer initialized")
    
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
            'time_series': {
                'trend_window': 4,
                'seasonality_periods': [4, 12],
                'anomaly_threshold': 2.5,
                'min_periods': 3
            },
            'statistics': {
                'significance_level': 0.05,
                'correlation_threshold': 0.3
            }
        }
    
    def prepare_time_series_data(self, df: pd.DataFrame, institution_col: str = 'bank', 
                                quarter_col: str = 'quarter') -> pd.DataFrame:
        """
        Prepare data for time series analysis by aggregating metrics by quarter
        
        Args:
            df: Input DataFrame with NLP analysis results
            institution_col: Column name for institution
            quarter_col: Column name for quarter
            
        Returns:
            DataFrame with time series metrics aggregated by quarter
        """
        # Convert quarter to datetime for proper time series handling
        df = df.copy()
        
        # Handle quarter format like 'Q1_2023'
        quarter_parts = df[quarter_col].str.extract(r'Q(\d)_(\d{4})')
        quarter_num = pd.to_numeric(quarter_parts[0], errors='coerce')
        year = pd.to_numeric(quarter_parts[1], errors='coerce')
        
        # Handle any parsing errors
        quarter_num = quarter_num.fillna(1).astype(int)
        year = year.fillna(2023).astype(int)
        
        # Create datetime: start of quarter
        # Calculate month for each quarter: Q1=1, Q2=4, Q3=7, Q4=10
        month = ((quarter_num - 1) * 3) + 1
        
        # Create date strings and convert to datetime
        date_strings = year.astype(str) + '-' + month.astype(str).str.zfill(2) + '-01'
        df['quarter_date'] = pd.to_datetime(date_strings)
        
        # Define aggregation metrics
        agg_metrics = {
            # Sentiment metrics
            'sentiment_positive': ['mean', 'std', 'count'],
            'sentiment_negative': ['mean', 'std', 'count'],
            'sentiment_neutral': ['mean', 'std', 'count'],
            'sentiment_confidence': ['mean', 'std'],
            
            # Risk metrics
            'risk_escalation_score': ['mean', 'std', 'max'],
            'stress_score': ['mean', 'std', 'max'],
            'confidence_score': ['mean', 'std', 'min'],
            
            # Tone metrics
            'hedging_score': ['mean', 'std', 'max'],
            'uncertainty_score': ['mean', 'std', 'max'],
            'formality_score': ['mean', 'std'],
            'complexity_score': ['mean', 'std']
        }
        
        # Aggregate by institution and quarter
        ts_data = df.groupby([institution_col, 'quarter_date']).agg(agg_metrics).reset_index()
        
        # Flatten column names
        ts_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in ts_data.columns]
        
        # Rename back the grouping columns
        ts_data = ts_data.rename(columns={
            f'{institution_col}_': institution_col,
            'quarter_date_': 'quarter_date'
        })
        
        # Sort by institution and date
        ts_data = ts_data.sort_values([institution_col, 'quarter_date']).reset_index(drop=True)
        
        logging.info(f"Prepared time series data: {len(ts_data)} records across {ts_data[institution_col].nunique()} institutions")
        
        return ts_data
    
    def detect_trends(self, ts_data: pd.DataFrame, institution: str, 
                     metrics: List[str]) -> Dict[str, Dict]:
        """
        Detect trends in time series metrics for a specific institution
        
        Args:
            ts_data: Time series data
            institution: Institution name
            metrics: List of metric columns to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        institution_data = ts_data[ts_data['bank'] == institution].copy()
        
        if len(institution_data) < self.min_periods:
            logging.warning(f"Insufficient data for trend analysis: {institution} ({len(institution_data)} periods)")
            return {}
        
        trend_results = {}
        
        for metric in metrics:
            if metric not in institution_data.columns:
                continue
                
            values = institution_data[metric].dropna()
            if len(values) < self.min_periods:
                continue
            
            # Calculate trend statistics
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend direction and significance
            trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            is_significant = p_value < self.significance_level
            
            # Calculate trend strength
            trend_strength = abs(r_value)
            
            # Calculate recent change (last vs previous period)
            recent_change = 0
            if len(values) >= 2:
                recent_change = (values.iloc[-1] - values.iloc[-2]) / abs(values.iloc[-2]) if values.iloc[-2] != 0 else 0
            
            trend_results[metric] = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': trend_direction,
                'is_significant': is_significant,
                'trend_strength': trend_strength,
                'recent_change_pct': recent_change * 100,
                'periods_analyzed': len(values),
                'current_value': values.iloc[-1] if len(values) > 0 else None,
                'mean_value': values.mean(),
                'std_value': values.std()
            }
        
        return trend_results
    
    def detect_seasonality(self, ts_data: pd.DataFrame, institution: str, 
                          metrics: List[str]) -> Dict[str, Dict]:
        """
        Detect seasonal patterns in time series metrics
        
        Args:
            ts_data: Time series data
            institution: Institution name
            metrics: List of metric columns to analyze
            
        Returns:
            Dictionary with seasonality analysis results
        """
        institution_data = ts_data[ts_data['bank'] == institution].copy()
        
        if len(institution_data) < max(self.seasonality_periods):
            logging.warning(f"Insufficient data for seasonality analysis: {institution}")
            return {}
        
        # Extract quarter from date
        institution_data['quarter_num'] = institution_data['quarter_date'].dt.quarter
        
        seasonality_results = {}
        
        for metric in metrics:
            if metric not in institution_data.columns:
                continue
                
            values = institution_data[metric].dropna()
            quarters = institution_data.loc[values.index, 'quarter_num']
            
            if len(values) < max(self.seasonality_periods):
                continue
            
            # Test for seasonal patterns using ANOVA
            quarter_groups = [values[quarters == q] for q in range(1, 5)]
            quarter_groups = [group for group in quarter_groups if len(group) > 0]
            
            if len(quarter_groups) < 2:
                continue
            
            try:
                f_stat, p_value = stats.f_oneway(*quarter_groups)
                is_seasonal = p_value < self.significance_level
                
                # Calculate quarterly statistics
                quarterly_stats = {}
                for q in range(1, 5):
                    q_data = values[quarters == q]
                    if len(q_data) > 0:
                        quarterly_stats[f'Q{q}'] = {
                            'mean': q_data.mean(),
                            'std': q_data.std(),
                            'count': len(q_data),
                            'min': q_data.min(),
                            'max': q_data.max()
                        }
                
                # Find peak and trough quarters
                q_means = {q: stats['mean'] for q, stats in quarterly_stats.items() if not np.isnan(stats['mean'])}
                peak_quarter = max(q_means.items(), key=lambda x: x[1])[0] if q_means else None
                trough_quarter = min(q_means.items(), key=lambda x: x[1])[0] if q_means else None
                
                seasonality_results[metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'is_seasonal': is_seasonal,
                    'quarterly_stats': quarterly_stats,
                    'peak_quarter': peak_quarter,
                    'trough_quarter': trough_quarter,
                    'seasonal_amplitude': max(q_means.values()) - min(q_means.values()) if q_means else 0
                }
                
            except Exception as e:
                logging.warning(f"Error in seasonality analysis for {metric}: {e}")
                continue
        
        return seasonality_results
    
    def detect_anomalies(self, ts_data: pd.DataFrame, institution: str, 
                        metrics: List[str]) -> Dict[str, Dict]:
        """
        Detect anomalies in time series using statistical methods
        
        Args:
            ts_data: Time series data
            institution: Institution name
            metrics: List of metric columns to analyze
            
        Returns:
            Dictionary with anomaly detection results
        """
        institution_data = ts_data[ts_data['bank'] == institution].copy()
        
        if len(institution_data) < self.min_periods:
            return {}
        
        anomaly_results = {}
        
        for metric in metrics:
            if metric not in institution_data.columns:
                continue
                
            values = institution_data[metric].dropna()
            dates = institution_data.loc[values.index, 'quarter_date']
            
            if len(values) < self.min_periods:
                continue
            
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(values))
            anomaly_mask = z_scores > self.anomaly_threshold
            
            # IQR based anomaly detection
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_anomaly_mask = (values < lower_bound) | (values > upper_bound)
            
            # Combine both methods
            combined_anomaly_mask = anomaly_mask | iqr_anomaly_mask
            
            anomalies = []
            if combined_anomaly_mask.any():
                anomaly_indices = np.where(combined_anomaly_mask)[0]
                for idx in anomaly_indices:
                    anomalies.append({
                        'date': dates.iloc[idx],
                        'value': values.iloc[idx],
                        'z_score': z_scores[idx],
                        'is_outlier_zscore': anomaly_mask[idx],
                        'is_outlier_iqr': iqr_anomaly_mask.iloc[idx],
                        'deviation_from_mean': values.iloc[idx] - values.mean(),
                        'percentile': stats.percentileofscore(values, values.iloc[idx])
                    })
            
            anomaly_results[metric] = {
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / len(values),
                'anomalies': anomalies,
                'mean_value': values.mean(),
                'std_value': values.std(),
                'min_value': values.min(),
                'max_value': values.max(),
                'iqr_bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
        
        return anomaly_results
    
    def calculate_correlations(self, ts_data: pd.DataFrame, institution: str, 
                             metrics: List[str]) -> Dict[str, float]:
        """
        Calculate correlations between different metrics
        
        Args:
            ts_data: Time series data
            institution: Institution name
            metrics: List of metric columns to analyze
            
        Returns:
            Dictionary with correlation coefficients
        """
        institution_data = ts_data[ts_data['bank'] == institution].copy()
        
        if len(institution_data) < self.min_periods:
            return {}
        
        # Select only the metrics that exist in the data
        available_metrics = [m for m in metrics if m in institution_data.columns]
        
        if len(available_metrics) < 2:
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = institution_data[available_metrics].corr()
        
        # Extract significant correlations
        correlations = {}
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:  # Avoid duplicates and self-correlations
                    corr_value = correlation_matrix.loc[metric1, metric2]
                    if not np.isnan(corr_value) and abs(corr_value) >= self.correlation_threshold:
                        correlations[f"{metric1}_vs_{metric2}"] = corr_value
        
        return correlations
    
    def generate_risk_signals(self, trend_results: Dict, anomaly_results: Dict, 
                            seasonality_results: Dict) -> Dict[str, Any]:
        """
        Generate risk signals based on time series analysis
        
        Args:
            trend_results: Results from trend analysis
            anomaly_results: Results from anomaly detection
            seasonality_results: Results from seasonality analysis
            
        Returns:
            Dictionary with risk signals and recommendations
        """
        risk_signals = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'recommendations': [],
            'key_metrics': {}
        }
        
        risk_score = 0
        
        # Analyze trends for risk signals
        for metric, trend_data in trend_results.items():
            if 'risk' in metric.lower() or 'stress' in metric.lower():
                if trend_data['trend_direction'] == 'increasing' and trend_data['is_significant']:
                    risk_score += 2
                    risk_signals['risk_factors'].append(f"Increasing trend in {metric}")
                    
            elif 'confidence' in metric.lower():
                if trend_data['trend_direction'] == 'decreasing' and trend_data['is_significant']:
                    risk_score += 1
                    risk_signals['risk_factors'].append(f"Decreasing confidence trend")
            
            elif 'negative' in metric.lower():
                if trend_data['trend_direction'] == 'increasing' and trend_data['is_significant']:
                    risk_score += 1
                    risk_signals['risk_factors'].append(f"Increasing negative sentiment")
        
        # Analyze anomalies for risk signals
        for metric, anomaly_data in anomaly_results.items():
            if anomaly_data['anomaly_rate'] > 0.2:  # More than 20% anomalies
                risk_score += 1
                risk_signals['risk_factors'].append(f"High anomaly rate in {metric}")
            
            # Check recent anomalies
            recent_anomalies = [a for a in anomaly_data['anomalies'] 
                              if (datetime.now() - a['date']).days < 180]  # Last 6 months
            if len(recent_anomalies) > 0:
                risk_score += 1
                risk_signals['risk_factors'].append(f"Recent anomalies detected in {metric}")
        
        # Determine overall risk level
        if risk_score >= 5:
            risk_signals['overall_risk_level'] = 'high'
        elif risk_score >= 3:
            risk_signals['overall_risk_level'] = 'medium'
        else:
            risk_signals['overall_risk_level'] = 'low'
        
        # Generate recommendations
        if risk_score > 0:
            risk_signals['recommendations'].append("Increase monitoring frequency")
            risk_signals['recommendations'].append("Review recent earnings calls for context")
            
        if risk_score >= 3:
            risk_signals['recommendations'].append("Conduct detailed risk assessment")
            risk_signals['recommendations'].append("Compare with peer institutions")
            
        if risk_score >= 5:
            risk_signals['recommendations'].append("Consider regulatory notification")
            risk_signals['recommendations'].append("Implement enhanced oversight measures")
        
        risk_signals['risk_score'] = risk_score
        
        return risk_signals
    
    def analyze_institution(self, ts_data: pd.DataFrame, institution: str) -> Dict[str, Any]:
        """
        Comprehensive time series analysis for a single institution
        
        Args:
            ts_data: Time series data
            institution: Institution name
            
        Returns:
            Complete analysis results
        """
        # Define key metrics to analyze
        key_metrics = [
            'sentiment_negative_mean', 'sentiment_positive_mean',
            'risk_escalation_score_mean', 'stress_score_mean', 'confidence_score_mean',
            'hedging_score_mean', 'uncertainty_score_mean'
        ]
        
        # Perform all analyses
        trend_results = self.detect_trends(ts_data, institution, key_metrics)
        seasonality_results = self.detect_seasonality(ts_data, institution, key_metrics)
        anomaly_results = self.detect_anomalies(ts_data, institution, key_metrics)
        correlations = self.calculate_correlations(ts_data, institution, key_metrics)
        risk_signals = self.generate_risk_signals(trend_results, anomaly_results, seasonality_results)
        
        return {
            'institution': institution,
            'analysis_date': datetime.now().isoformat(),
            'data_periods': len(ts_data[ts_data['bank'] == institution]),
            'trends': trend_results,
            'seasonality': seasonality_results,
            'anomalies': anomaly_results,
            'correlations': correlations,
            'risk_signals': risk_signals
        }

def get_time_series_analyzer(config_path: Optional[str] = None) -> TimeSeriesAnalyzer:
    """Get time series analyzer instance"""
    return TimeSeriesAnalyzer(config_path)