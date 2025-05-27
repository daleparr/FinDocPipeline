"""
Advanced Anomaly Detection Engine for Financial Risk Monitoring
Provides multi-method anomaly detection with ensemble scoring
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Advanced multi-method anomaly detection for financial risk monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Anomaly detection configuration
        self.contamination = self.config.get('anomaly_detection', {}).get('contamination', 0.1)
        self.z_threshold = self.config.get('anomaly_detection', {}).get('z_threshold', 2.5)
        self.iqr_multiplier = self.config.get('anomaly_detection', {}).get('iqr_multiplier', 1.5)
        self.ensemble_threshold = self.config.get('anomaly_detection', {}).get('ensemble_threshold', 0.6)
        
        # Method weights for ensemble
        self.method_weights = self.config.get('anomaly_detection', {}).get('method_weights', {
            'isolation_forest': 0.25,
            'elliptic_envelope': 0.25,
            'z_score': 0.2,
            'iqr': 0.15,
            'dbscan': 0.15
        })
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.elliptic_envelope = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        logging.info("Anomaly detector initialized with ensemble methods")
    
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
            'anomaly_detection': {
                'contamination': 0.1,
                'z_threshold': 2.5,
                'iqr_multiplier': 1.5,
                'ensemble_threshold': 0.6,
                'method_weights': {
                    'isolation_forest': 0.25,
                    'elliptic_envelope': 0.25,
                    'z_score': 0.2,
                    'iqr': 0.15,
                    'dbscan': 0.15
                }
            }
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for anomaly detection
        
        Args:
            df: Input DataFrame with NLP analysis results
            
        Returns:
            Tuple of (feature_df, feature_names)
        """
        # Select numerical features for anomaly detection
        feature_columns = [
            'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'sentiment_confidence', 'risk_escalation_score', 'stress_score',
            'confidence_score', 'hedging_score', 'uncertainty_score',
            'formality_score', 'complexity_score'
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No suitable features found for anomaly detection")
        
        # Create feature matrix
        feature_df = df[available_features].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        # Add derived features
        if 'sentiment_positive' in feature_df.columns and 'sentiment_negative' in feature_df.columns:
            feature_df['sentiment_polarity'] = feature_df['sentiment_positive'] - feature_df['sentiment_negative']
        
        if 'risk_escalation_score' in feature_df.columns and 'confidence_score' in feature_df.columns:
            feature_df['risk_confidence_ratio'] = feature_df['risk_escalation_score'] / (feature_df['confidence_score'] + 0.001)
        
        if 'hedging_score' in feature_df.columns and 'uncertainty_score' in feature_df.columns:
            feature_df['uncertainty_index'] = feature_df['hedging_score'] + feature_df['uncertainty_score']
        
        # Update feature names
        feature_names = list(feature_df.columns)
        
        logging.info(f"Prepared {len(feature_names)} features for anomaly detection")
        
        return feature_df, feature_names
    
    def detect_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        try:
            # Fit and predict
            anomaly_labels = self.isolation_forest.fit_predict(X)
            anomaly_scores = self.isolation_forest.decision_function(X)
            
            # Convert to binary (1 = normal, -1 = anomaly)
            anomaly_binary = (anomaly_labels == -1).astype(int)
            
            # Normalize scores to [0, 1] range
            normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            return anomaly_binary, normalized_scores
            
        except Exception as e:
            logging.warning(f"Isolation Forest failed: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def detect_elliptic_envelope(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Elliptic Envelope
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        try:
            # Fit and predict
            anomaly_labels = self.elliptic_envelope.fit_predict(X)
            
            # Get Mahalanobis distances as scores
            anomaly_scores = self.elliptic_envelope.mahalanobis(X)
            
            # Convert to binary
            anomaly_binary = (anomaly_labels == -1).astype(int)
            
            # Normalize scores
            normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            return anomaly_binary, normalized_scores
            
        except Exception as e:
            logging.warning(f"Elliptic Envelope failed: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def detect_z_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        try:
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(X, axis=0))
            
            # Max Z-score across features for each sample
            max_z_scores = np.max(z_scores, axis=1)
            
            # Binary anomaly detection
            anomaly_binary = (max_z_scores > self.z_threshold).astype(int)
            
            # Normalize scores
            normalized_scores = max_z_scores / (self.z_threshold * 2)  # Normalize around threshold
            normalized_scores = np.clip(normalized_scores, 0, 1)
            
            return anomaly_binary, normalized_scores
            
        except Exception as e:
            logging.warning(f"Z-score method failed: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def detect_iqr(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using IQR method
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        try:
            anomaly_binary = np.zeros(len(X))
            anomaly_scores = np.zeros(len(X))
            
            for i in range(X.shape[1]):
                feature = X[:, i]
                q1, q3 = np.percentile(feature, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - self.iqr_multiplier * iqr
                upper_bound = q3 + self.iqr_multiplier * iqr
                
                # Check for outliers
                outliers = (feature < lower_bound) | (feature > upper_bound)
                anomaly_binary = np.maximum(anomaly_binary, outliers.astype(int))
                
                # Calculate distance from bounds
                distances = np.maximum(
                    np.maximum(lower_bound - feature, 0),
                    np.maximum(feature - upper_bound, 0)
                )
                normalized_distances = distances / (iqr + 0.001)
                anomaly_scores = np.maximum(anomaly_scores, normalized_distances)
            
            # Normalize final scores
            if anomaly_scores.max() > 0:
                anomaly_scores = anomaly_scores / anomaly_scores.max()
            
            return anomaly_binary, anomaly_scores
            
        except Exception as e:
            logging.warning(f"IQR method failed: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def detect_dbscan(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using DBSCAN clustering
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        try:
            # Fit DBSCAN
            cluster_labels = self.dbscan.fit_predict(X)
            
            # Points with label -1 are considered anomalies
            anomaly_binary = (cluster_labels == -1).astype(int)
            
            # Calculate distance to nearest cluster center as score
            anomaly_scores = np.zeros(len(X))
            
            # For each point, calculate distance to nearest cluster center
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
            
            if len(unique_clusters) > 0:
                for i, point in enumerate(X):
                    if cluster_labels[i] == -1:  # Anomaly
                        min_distance = float('inf')
                        for cluster_id in unique_clusters:
                            cluster_points = X[cluster_labels == cluster_id]
                            cluster_center = np.mean(cluster_points, axis=0)
                            distance = np.linalg.norm(point - cluster_center)
                            min_distance = min(min_distance, distance)
                        anomaly_scores[i] = min_distance
                
                # Normalize scores
                if anomaly_scores.max() > 0:
                    anomaly_scores = anomaly_scores / anomaly_scores.max()
            
            return anomaly_binary, anomaly_scores
            
        except Exception as e:
            logging.warning(f"DBSCAN method failed: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def ensemble_detection(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform ensemble anomaly detection using multiple methods
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with results from all methods and ensemble
        """
        # Scale data for methods that require it
        X_standard = self.standard_scaler.fit_transform(X)
        X_robust = self.robust_scaler.fit_transform(X)
        
        # Apply different methods
        methods_results = {}
        
        # Isolation Forest (uses original scale)
        if_binary, if_scores = self.detect_isolation_forest(X)
        methods_results['isolation_forest'] = {'binary': if_binary, 'scores': if_scores}
        
        # Elliptic Envelope (uses standardized data)
        ee_binary, ee_scores = self.detect_elliptic_envelope(X_standard)
        methods_results['elliptic_envelope'] = {'binary': ee_binary, 'scores': ee_scores}
        
        # Z-score (uses standardized data)
        z_binary, z_scores = self.detect_z_score(X_standard)
        methods_results['z_score'] = {'binary': z_binary, 'scores': z_scores}
        
        # IQR (uses original scale)
        iqr_binary, iqr_scores = self.detect_iqr(X)
        methods_results['iqr'] = {'binary': iqr_binary, 'scores': iqr_scores}
        
        # DBSCAN (uses robust scaled data)
        db_binary, db_scores = self.detect_dbscan(X_robust)
        methods_results['dbscan'] = {'binary': db_binary, 'scores': db_scores}
        
        # Calculate ensemble scores
        ensemble_scores = np.zeros(len(X))
        ensemble_binary = np.zeros(len(X))
        
        for method, weight in self.method_weights.items():
            if method in methods_results:
                ensemble_scores += weight * methods_results[method]['scores']
                ensemble_binary += weight * methods_results[method]['binary']
        
        # Final ensemble decision
        final_binary = (ensemble_binary >= self.ensemble_threshold).astype(int)
        
        methods_results['ensemble'] = {
            'binary': final_binary,
            'scores': ensemble_scores,
            'weighted_votes': ensemble_binary
        }
        
        return methods_results
    
    def analyze_anomalies(self, df: pd.DataFrame, feature_names: List[str], 
                         anomaly_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Analyze detected anomalies and add contextual information
        
        Args:
            df: Original DataFrame
            feature_names: List of feature names
            anomaly_results: Results from ensemble detection
            
        Returns:
            DataFrame with anomaly analysis
        """
        # Create results DataFrame
        results_df = df.copy()
        
        # Add anomaly detection results
        for method, results in anomaly_results.items():
            results_df[f'anomaly_{method}'] = results['binary']
            results_df[f'anomaly_score_{method}'] = results['scores']
        
        # Add ensemble results
        results_df['is_anomaly'] = anomaly_results['ensemble']['binary']
        results_df['anomaly_score'] = anomaly_results['ensemble']['scores']
        results_df['anomaly_confidence'] = anomaly_results['ensemble']['weighted_votes']
        
        # Calculate anomaly severity
        results_df['anomaly_severity'] = pd.cut(
            results_df['anomaly_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        # Add contextual features for anomalies
        anomaly_mask = results_df['is_anomaly'] == 1
        
        if anomaly_mask.any():
            # Calculate feature contributions to anomaly
            feature_df, _ = self.prepare_features(df)
            X = feature_df.values
            
            # Z-scores for feature contribution analysis
            z_scores = np.abs(stats.zscore(X, axis=0))
            
            # For each anomaly, find the most contributing features
            anomaly_features = []
            for idx in results_df[anomaly_mask].index:
                row_idx = results_df.index.get_loc(idx)
                feature_z_scores = z_scores[row_idx]
                top_features = np.argsort(feature_z_scores)[-3:][::-1]  # Top 3 features
                contributing_features = [feature_names[i] for i in top_features]
                anomaly_features.append(contributing_features)
            
            results_df.loc[anomaly_mask, 'contributing_features'] = [
                ', '.join(features) for features in anomaly_features
            ]
        
        return results_df
    
    def generate_anomaly_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive anomaly detection report
        
        Args:
            results_df: DataFrame with anomaly analysis results
            
        Returns:
            Dictionary with anomaly report
        """
        total_records = len(results_df)
        anomalies = results_df[results_df['is_anomaly'] == 1]
        
        report = {
            'summary': {
                'total_records': total_records,
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / total_records if total_records > 0 else 0,
                'analysis_date': datetime.now().isoformat()
            },
            'severity_distribution': {},
            'method_agreement': {},
            'top_anomalies': [],
            'feature_analysis': {},
            'temporal_patterns': {}
        }
        
        if len(anomalies) > 0:
            # Severity distribution
            # Convert categorical to string to avoid issues
            severity_counts = anomalies['anomaly_severity'].astype(str).value_counts()
            report['severity_distribution'] = severity_counts.to_dict()
            
            # Method agreement analysis
            method_columns = [col for col in results_df.columns if col.startswith('anomaly_') and not col.startswith('anomaly_score')]
            if method_columns:
                # Convert any categorical columns to numeric before summing
                method_data = results_df[method_columns].copy()
                for col in method_columns:
                    if method_data[col].dtype.name == 'category':
                        method_data[col] = pd.to_numeric(method_data[col], errors='coerce').fillna(0).astype(int)
                    elif method_data[col].dtype == 'object':
                        method_data[col] = pd.to_numeric(method_data[col], errors='coerce').fillna(0).astype(int)
                method_agreement = method_data.sum()
                report['method_agreement'] = method_agreement.to_dict()
            
            # Top anomalies by score
            top_anomalies = anomalies.nlargest(10, 'anomaly_score')
            report['top_anomalies'] = []
            
            for _, row in top_anomalies.iterrows():
                anomaly_info = {
                    'index': int(row.name),
                    'anomaly_score': float(row['anomaly_score']),
                    'severity': str(row['anomaly_severity']),
                    'contributing_features': str(row.get('contributing_features', '')),
                }
                
                # Add context if available
                if 'bank' in row:
                    anomaly_info['institution'] = str(row['bank'])
                if 'quarter' in row:
                    anomaly_info['quarter'] = str(row['quarter'])
                if 'speaker_norm' in row:
                    anomaly_info['speaker'] = str(row['speaker_norm'])
                if 'text' in row:
                    anomaly_info['text_preview'] = str(row['text'])[:100] + '...' if len(str(row['text'])) > 100 else str(row['text'])
                
                report['top_anomalies'].append(anomaly_info)
            
            # Feature analysis
            if 'contributing_features' in anomalies.columns:
                all_features = []
                for features_str in anomalies['contributing_features'].dropna():
                    all_features.extend(features_str.split(', '))
                
                from collections import Counter
                feature_counts = Counter(all_features)
                report['feature_analysis'] = dict(feature_counts.most_common(10))
            
            # Temporal patterns (if date information available)
            if 'quarter' in anomalies.columns:
                temporal_counts = anomalies['quarter'].value_counts()
                report['temporal_patterns'] = temporal_counts.to_dict()
        
        return report
    
    def detect_anomalies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main method to detect anomalies in the dataset
        
        Args:
            df: Input DataFrame with NLP analysis results
            
        Returns:
            Tuple of (results_df, anomaly_report)
        """
        logging.info(f"Starting anomaly detection on {len(df)} records")
        
        # Prepare features
        feature_df, feature_names = self.prepare_features(df)
        X = feature_df.values
        
        # Perform ensemble detection
        anomaly_results = self.ensemble_detection(X)
        
        # Analyze results
        results_df = self.analyze_anomalies(df, feature_names, anomaly_results)
        
        # Generate report
        report = self.generate_anomaly_report(results_df)
        
        logging.info(f"Anomaly detection complete: {report['summary']['total_anomalies']} anomalies detected")
        
        return results_df, report

def get_anomaly_detector(config_path: Optional[str] = None) -> AnomalyDetector:
    """Get anomaly detector instance"""
    return AnomalyDetector(config_path)