"""
G-SIB Monitor for BoE Mosaic Lens
Tracks Global Systemically Important Banks with emphasis on cross-market correlations
and systemic risk indicators for BoE supervisory oversight
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class GSIBMonitor:
    """
    Monitor for Global Systemically Important Banks
    Focuses on cross-market correlations and systemic risk detection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize G-SIB monitor
        
        Args:
            config_path: Path to G-SIB institutions configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.gsib_institutions = self._load_gsib_institutions()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load G-SIB monitoring configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "gsib_institutions.yaml"
        
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_gsib_institutions(self) -> Dict[str, Dict]:
        """Load G-SIB institution mappings from bucket structure"""
        institutions = {}
        gsib_data = self.config.get('global_gsibs', {})
        
        for bucket_or_region, banks in gsib_data.items():
            if isinstance(banks, list):
                for bank in banks:
                    institutions[bank['ticker']] = {
                        'name': bank['name'],
                        'full_name': bank['full_name'],
                        'country': bank['country'],
                        'systemic_weight': bank['systemic_weight'],
                        'market_cap_tier': bank['market_cap_tier'],
                        'fsb_bucket': bank.get('fsb_bucket', bucket_or_region),
                        'sector': bank.get('sector', 'Investment Banking & Brokerage'),
                        'earnings_frequency': bank.get('earnings_frequency', 'quarterly'),
                        'typical_earnings_months': bank.get('typical_earnings_months', [1, 4, 7, 10])
                    }
        
        return institutions
    
    def get_gsib_tickers(self, region: Optional[str] = None, bucket: Optional[str] = None) -> List[str]:
        """Get list of G-SIB tickers, optionally filtered by region or FSB bucket"""
        if bucket:
            return [ticker for ticker, info in self.gsib_institutions.items()
                   if info.get('fsb_bucket') == bucket]
        elif region:
            return [ticker for ticker, info in self.gsib_institutions.items()
                   if info.get('country') == region or info.get('fsb_bucket') == region]
        return list(self.gsib_institutions.keys())
    
    def get_gsib_by_systemic_importance(self) -> Dict[str, List[str]]:
        """Get G-SIBs organized by FSB systemic importance buckets"""
        buckets = {}
        
        for ticker, info in self.gsib_institutions.items():
            bucket = info.get('fsb_bucket', 'unknown')
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(ticker)
        
        return buckets
    
    def get_total_gsib_count(self) -> int:
        """Get total number of G-SIBs being monitored"""
        return len(self.gsib_institutions)
    
    def track_global_gsib_movements(self, period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """
        Track market movements across all G-SIB institutions
        
        Args:
            period: Data period for analysis
            
        Returns:
            Dictionary mapping ticker to market data
        """
        try:
            from .yahoo_finance_client import get_yahoo_finance_client
            from .market_indicators import get_market_indicators_engine
            
            client = get_yahoo_finance_client()
            indicators_engine = get_market_indicators_engine()
            
            tickers = self.get_gsib_tickers()
            market_data = {}
            
            self.logger.info(f"Tracking {len(tickers)} G-SIB institutions")
            
            for ticker in tickers:
                try:
                    # Fetch market data
                    data = client.fetch_market_data(ticker, period=period)
                    
                    if not data.empty:
                        # Add comprehensive indicators
                        enhanced_data = indicators_engine.calculate_comprehensive_indicators(data)
                        
                        # Add G-SIB specific metadata
                        enhanced_data['gsib_info'] = str(self.gsib_institutions[ticker])
                        enhanced_data['systemic_weight'] = self.gsib_institutions[ticker]['systemic_weight']
                        enhanced_data['region'] = self.gsib_institutions[ticker]['country']
                        
                        market_data[ticker] = enhanced_data
                        
                        self.logger.info(f"Successfully tracked {ticker}")
                    else:
                        self.logger.warning(f"No data available for {ticker}")
                        
                except Exception as e:
                    self.logger.error(f"Error tracking {ticker}: {e}")
                    continue
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error in global G-SIB tracking: {e}")
            return {}
    
    def detect_cross_market_correlations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detect cross-market correlations and interconnectedness patterns
        
        Args:
            market_data: Dictionary mapping ticker to market data
            
        Returns:
            Dictionary with correlation analysis results
        """
        if not market_data:
            return {}
        
        try:
            # Extract returns data
            returns_data = {}
            for ticker, df in market_data.items():
                if not df.empty and 'daily_return' in df.columns:
                    returns_data[ticker] = df['daily_return'].dropna()
            
            if len(returns_data) < 2:
                self.logger.warning("Insufficient data for correlation analysis")
                return {}
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Check if we have sufficient data
            if returns_df.empty or len(returns_df.columns) < 2:
                return {}
            
            # Calculate correlation matrix (pandas corr() handles NaN values automatically)
            correlation_matrix = returns_df.corr(min_periods=2)
            
            # Remove any columns/rows that are all NaN
            correlation_matrix = correlation_matrix.dropna(how='all').dropna(axis=1, how='all')
            
            if correlation_matrix.empty:
                return {}
            
            # Analyze correlation patterns
            analysis = {
                'correlation_matrix': correlation_matrix,
                'summary_stats': self._analyze_correlation_matrix(correlation_matrix),
                'regional_correlations': self._analyze_regional_correlations(correlation_matrix, returns_data),
                'systemic_clusters': self._identify_systemic_clusters(correlation_matrix),
                'contagion_risk': self._assess_contagion_risk(correlation_matrix),
                'network_analysis': self._perform_network_analysis(correlation_matrix)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in cross-market correlation analysis: {e}")
            return {}
    
    def _analyze_correlation_matrix(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlation matrix for key statistics"""
        # Remove diagonal (self-correlations)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        correlations = upper_triangle.stack().dropna()
        
        return {
            'mean_correlation': correlations.mean(),
            'median_correlation': correlations.median(),
            'std_correlation': correlations.std(),
            'max_correlation': correlations.max(),
            'min_correlation': correlations.min(),
            'high_correlation_pairs': (correlations > 0.7).sum(),
            'negative_correlation_pairs': (correlations < -0.3).sum(),
            'total_pairs': len(correlations)
        }
    
    def _analyze_regional_correlations(self, corr_matrix: pd.DataFrame, returns_data: Dict) -> Dict[str, Dict]:
        """Analyze correlations within and across regions"""
        regional_analysis = {}
        
        # Group tickers by region
        regions = {}
        for ticker in returns_data.keys():
            if ticker in self.gsib_institutions:
                region = self.gsib_institutions[ticker]['country']
                if region not in regions:
                    regions[region] = []
                regions[region].append(ticker)
        
        # Calculate within-region and cross-region correlations
        for region1, tickers1 in regions.items():
            regional_analysis[region1] = {}
            
            # Within-region correlations
            if len(tickers1) > 1:
                within_region_corrs = []
                for i, ticker1 in enumerate(tickers1):
                    for ticker2 in tickers1[i+1:]:
                        if ticker1 in corr_matrix.index and ticker2 in corr_matrix.columns:
                            within_region_corrs.append(corr_matrix.loc[ticker1, ticker2])
                
                if within_region_corrs:
                    regional_analysis[region1]['within_region'] = {
                        'mean': np.mean(within_region_corrs),
                        'std': np.std(within_region_corrs),
                        'max': np.max(within_region_corrs),
                        'count': len(within_region_corrs)
                    }
            
            # Cross-region correlations
            for region2, tickers2 in regions.items():
                if region1 != region2:
                    cross_region_corrs = []
                    for ticker1 in tickers1:
                        for ticker2 in tickers2:
                            if ticker1 in corr_matrix.index and ticker2 in corr_matrix.columns:
                                cross_region_corrs.append(corr_matrix.loc[ticker1, ticker2])
                    
                    if cross_region_corrs:
                        regional_analysis[region1][f'cross_region_{region2}'] = {
                            'mean': np.mean(cross_region_corrs),
                            'std': np.std(cross_region_corrs),
                            'max': np.max(cross_region_corrs),
                            'count': len(cross_region_corrs)
                        }
        
        return regional_analysis
    
    def _identify_systemic_clusters(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Identify clusters of highly correlated institutions"""
        try:
            # Clean correlation matrix - handle NaN and infinite values
            clean_corr_matrix = corr_matrix.fillna(0)  # Replace NaN with 0
            clean_corr_matrix = clean_corr_matrix.replace([np.inf, -np.inf], 0)  # Replace inf with 0
            
            # Ensure diagonal is 1 (perfect self-correlation)
            np.fill_diagonal(clean_corr_matrix.values, 1.0)
            
            # Use correlation matrix for clustering
            distance_matrix = 1 - np.abs(clean_corr_matrix)
            
            # Ensure distance matrix is valid (finite values, non-negative)
            distance_matrix = np.clip(distance_matrix, 0, 2)  # Distance should be between 0 and 2
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=2.0, neginf=0.0)
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert to condensed distance matrix
            try:
                condensed_distances = squareform(distance_matrix, checks=True)
            except ValueError as e:
                self.logger.warning(f"Distance matrix validation failed: {e}, using fallback")
                # Fallback: ensure matrix is symmetric and valid
                distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
                np.fill_diagonal(distance_matrix, 0)  # Diagonal should be 0 for distance
                condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Get clusters (using different numbers of clusters)
            clusters_2 = fcluster(linkage_matrix, 2, criterion='maxclust')
            clusters_3 = fcluster(linkage_matrix, 3, criterion='maxclust')
            clusters_4 = fcluster(linkage_matrix, 4, criterion='maxclust')
            
            # Organize results
            tickers = corr_matrix.index.tolist()
            
            clustering_results = {
                '2_clusters': {f'cluster_{i}': [tickers[j] for j, c in enumerate(clusters_2) if c == i] 
                              for i in range(1, 3)},
                '3_clusters': {f'cluster_{i}': [tickers[j] for j, c in enumerate(clusters_3) if c == i] 
                              for i in range(1, 4)},
                '4_clusters': {f'cluster_{i}': [tickers[j] for j, c in enumerate(clusters_4) if c == i] 
                              for i in range(1, 5)},
                'linkage_matrix': linkage_matrix.tolist()
            }
            
            return clustering_results
            
        except Exception as e:
            self.logger.error(f"Error in systemic clustering: {e}")
            return {}
    
    def _assess_contagion_risk(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Assess contagion risk based on correlation patterns"""
        try:
            # Calculate contagion risk metrics
            correlations = corr_matrix.values
            np.fill_diagonal(correlations, 0)  # Remove self-correlations
            
            # Average correlation for each institution
            avg_correlations = np.mean(np.abs(correlations), axis=1)
            
            # Maximum correlation for each institution
            max_correlations = np.max(np.abs(correlations), axis=1)
            
            # Number of high correlations (>0.7) for each institution
            high_corr_counts = np.sum(np.abs(correlations) > 0.7, axis=1)
            
            # Create contagion risk scores
            tickers = corr_matrix.index.tolist()
            contagion_scores = {}
            
            for i, ticker in enumerate(tickers):
                # Weight by systemic importance
                systemic_weight = self.gsib_institutions.get(ticker, {}).get('systemic_weight', 0.1)
                
                contagion_score = (
                    avg_correlations[i] * 0.4 +
                    max_correlations[i] * 0.3 +
                    (high_corr_counts[i] / len(tickers)) * 0.3
                ) * systemic_weight
                
                contagion_scores[ticker] = {
                    'contagion_score': contagion_score,
                    'avg_correlation': avg_correlations[i],
                    'max_correlation': max_correlations[i],
                    'high_corr_count': high_corr_counts[i],
                    'systemic_weight': systemic_weight
                }
            
            # Overall system contagion risk
            system_contagion_risk = np.mean(list(score['contagion_score'] for score in contagion_scores.values()))
            
            # Identify most systemically risky institutions
            sorted_scores = sorted(contagion_scores.items(), key=lambda x: x[1]['contagion_score'], reverse=True)
            
            return {
                'individual_scores': contagion_scores,
                'system_contagion_risk': system_contagion_risk,
                'highest_risk_institutions': sorted_scores[:5],
                'risk_level': self._classify_contagion_risk(system_contagion_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing contagion risk: {e}")
            return {}
    
    def _classify_contagion_risk(self, risk_score: float) -> str:
        """Classify contagion risk level"""
        if risk_score > 0.8:
            return "CRITICAL"
        elif risk_score > 0.6:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _perform_network_analysis(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform network analysis on correlation matrix"""
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for ticker in corr_matrix.index:
                G.add_node(ticker, **self.gsib_institutions.get(ticker, {}))
            
            # Add edges for significant correlations
            threshold = 0.5  # Minimum correlation for edge
            for i, ticker1 in enumerate(corr_matrix.index):
                for j, ticker2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicate edges
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > threshold:
                            G.add_edge(ticker1, ticker2, weight=abs(corr_value), correlation=corr_value)
            
            # Calculate network metrics
            network_metrics = {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'centrality_measures': {
                    'degree_centrality': nx.degree_centrality(G),
                    'betweenness_centrality': nx.betweenness_centrality(G),
                    'closeness_centrality': nx.closeness_centrality(G),
                    'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=1000)
                }
            }
            
            # Identify most central institutions
            degree_central = sorted(network_metrics['centrality_measures']['degree_centrality'].items(), 
                                  key=lambda x: x[1], reverse=True)
            betweenness_central = sorted(network_metrics['centrality_measures']['betweenness_centrality'].items(), 
                                       key=lambda x: x[1], reverse=True)
            
            network_metrics['most_central_institutions'] = {
                'by_degree': degree_central[:5],
                'by_betweenness': betweenness_central[:5]
            }
            
            return network_metrics
            
        except Exception as e:
            self.logger.error(f"Error in network analysis: {e}")
            return {}
    
    def calculate_systemic_risk_score(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate comprehensive systemic risk score
        
        Args:
            market_data: Dictionary mapping ticker to market data
            
        Returns:
            Dictionary with systemic risk assessment
        """
        if not market_data:
            return {'systemic_risk_score': 0, 'risk_level': 'LOW'}
        
        try:
            # Get correlation analysis
            correlation_analysis = self.detect_cross_market_correlations(market_data)
            
            if not correlation_analysis:
                return {'systemic_risk_score': 0, 'risk_level': 'LOW'}
            
            # Extract key metrics
            corr_stats = correlation_analysis.get('summary_stats', {})
            contagion_risk = correlation_analysis.get('contagion_risk', {})
            
            # Calculate individual risk components
            correlation_risk = min(corr_stats.get('mean_correlation', 0) * 2, 1.0)  # Scale to 0-1
            contagion_risk_score = contagion_risk.get('system_contagion_risk', 0)
            
            # Market stress indicators
            stress_indicators = self._calculate_market_stress_indicators(market_data)
            market_stress_score = stress_indicators.get('overall_stress_score', 0)
            
            # Volatility clustering
            volatility_clustering = self._assess_volatility_clustering(market_data)
            vol_clustering_score = volatility_clustering.get('clustering_score', 0)
            
            # Weighted systemic risk score
            systemic_risk_score = (
                correlation_risk * 0.3 +
                contagion_risk_score * 0.3 +
                market_stress_score * 0.25 +
                vol_clustering_score * 0.15
            )
            
            # Classify risk level
            if systemic_risk_score > 0.8:
                risk_level = "CRITICAL"
            elif systemic_risk_score > 0.6:
                risk_level = "HIGH"
            elif systemic_risk_score > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'systemic_risk_score': systemic_risk_score,
                'risk_level': risk_level,
                'component_scores': {
                    'correlation_risk': correlation_risk,
                    'contagion_risk': contagion_risk_score,
                    'market_stress': market_stress_score,
                    'volatility_clustering': vol_clustering_score
                },
                'detailed_analysis': {
                    'correlation_analysis': correlation_analysis,
                    'stress_indicators': stress_indicators,
                    'volatility_clustering': volatility_clustering
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating systemic risk score: {e}")
            return {'systemic_risk_score': 0, 'risk_level': 'LOW', 'error': str(e)}
    
    def _calculate_market_stress_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate market stress indicators across G-SIBs"""
        stress_metrics = {}
        
        try:
            # Collect stress indicators from all institutions
            all_returns = []
            all_volatilities = []
            negative_return_counts = []
            high_vol_counts = []
            
            for ticker, df in market_data.items():
                if df.empty:
                    continue
                
                if 'daily_return' in df.columns:
                    returns = df['daily_return'].dropna()
                    all_returns.extend(returns.tolist())
                    negative_return_counts.append((returns < -0.02).sum())  # >2% decline
                
                if 'volatility_20d' in df.columns:
                    volatilities = df['volatility_20d'].dropna()
                    all_volatilities.extend(volatilities.tolist())
                    vol_threshold = volatilities.quantile(0.9) if len(volatilities) > 0 else 0.02
                    high_vol_counts.append((volatilities > vol_threshold).sum())
            
            if all_returns:
                stress_metrics['return_stress'] = {
                    'mean_return': np.mean(all_returns),
                    'return_volatility': np.std(all_returns),
                    'negative_return_ratio': sum(negative_return_counts) / len(market_data),
                    'tail_risk': np.percentile(all_returns, 5)  # 5th percentile (VaR)
                }
            
            if all_volatilities:
                stress_metrics['volatility_stress'] = {
                    'mean_volatility': np.mean(all_volatilities),
                    'volatility_of_volatility': np.std(all_volatilities),
                    'high_vol_ratio': sum(high_vol_counts) / len(market_data)
                }
            
            # Calculate overall stress score
            return_stress_score = 0
            if 'return_stress' in stress_metrics:
                rs = stress_metrics['return_stress']
                return_stress_score = (
                    max(0, -rs['mean_return'] * 10) +  # Negative returns increase stress
                    min(1, rs['return_volatility'] * 5) +  # High volatility increases stress
                    min(1, rs['negative_return_ratio'] / 5)  # Many negative days increase stress
                ) / 3
            
            vol_stress_score = 0
            if 'volatility_stress' in stress_metrics:
                vs = stress_metrics['volatility_stress']
                vol_stress_score = (
                    min(1, vs['mean_volatility'] * 20) +  # High volatility increases stress
                    min(1, vs['volatility_of_volatility'] * 50) +  # Volatile volatility increases stress
                    min(1, vs['high_vol_ratio'] / 3)  # Many high-vol days increase stress
                ) / 3
            
            overall_stress_score = (return_stress_score + vol_stress_score) / 2
            stress_metrics['overall_stress_score'] = overall_stress_score
            
            return stress_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating market stress indicators: {e}")
            return {'overall_stress_score': 0}
    
    def _assess_volatility_clustering(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess volatility clustering across G-SIBs"""
        try:
            clustering_metrics = {}
            
            # Collect volatility data
            vol_data = {}
            for ticker, df in market_data.items():
                if not df.empty and 'volatility_20d' in df.columns:
                    vol_data[ticker] = df['volatility_20d'].dropna()
            
            if len(vol_data) < 2:
                return {'clustering_score': 0}
            
            # Create volatility DataFrame
            vol_df = pd.DataFrame(vol_data)
            vol_df = vol_df.dropna()
            
            if vol_df.empty:
                return {'clustering_score': 0}
            
            # Calculate volatility correlations
            vol_correlations = vol_df.corr()
            
            # Assess clustering
            mask = np.triu(np.ones_like(vol_correlations, dtype=bool), k=1)
            upper_triangle = vol_correlations.where(mask)
            correlations = upper_triangle.stack().dropna()
            
            clustering_score = correlations.mean() if len(correlations) > 0 else 0
            
            clustering_metrics = {
                'clustering_score': max(0, clustering_score),  # Ensure non-negative
                'mean_vol_correlation': correlations.mean(),
                'high_clustering_pairs': (correlations > 0.7).sum(),
                'total_pairs': len(correlations)
            }
            
            return clustering_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing volatility clustering: {e}")
            return {'clustering_score': 0}
    
    def generate_contagion_alerts(self, correlation_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate contagion alerts based on correlation analysis
        
        Args:
            correlation_analysis: Results from cross-market correlation analysis
            
        Returns:
            List of contagion alerts
        """
        alerts = []
        
        try:
            # Check for high system-wide correlations
            corr_stats = correlation_analysis.get('summary_stats', {})
            mean_corr = corr_stats.get('mean_correlation', 0)
            high_corr_pairs = corr_stats.get('high_correlation_pairs', 0)
            
            if mean_corr > 0.7:
                alerts.append({
                    'alert_type': 'HIGH_SYSTEM_CORRELATION',
                    'severity': 'HIGH',
                    'message': f"System-wide correlation elevated at {mean_corr:.3f}",
                    'metric_value': mean_corr,
                    'threshold': 0.7,
                    'timestamp': datetime.now()
                })
            
            if high_corr_pairs > 5:
                alerts.append({
                    'alert_type': 'MULTIPLE_HIGH_CORRELATIONS',
                    'severity': 'MEDIUM',
                    'message': f"{high_corr_pairs} institution pairs showing high correlation (>0.7)",
                    'metric_value': high_corr_pairs,
                    'threshold': 5,
                    'timestamp': datetime.now()
                })
            
            # Check contagion risk
            contagion_risk = correlation_analysis.get('contagion_risk', {})
            system_risk = contagion_risk.get('system_contagion_risk', 0)
            risk_level = contagion_risk.get('risk_level', 'LOW')
            
            if risk_level in ['HIGH', 'CRITICAL']:
                alerts.append({
                    'alert_type': 'SYSTEMIC_CONTAGION_RISK',
                    'severity': risk_level,
                    'message': f"Systemic contagion risk elevated: {risk_level} ({system_risk:.3f})",
                    'metric_value': system_risk,
                    'threshold': 0.6,
                    'timestamp': datetime.now()
                })
            
            # Check for specific high-risk institutions
            highest_risk = contagion_risk.get('highest_risk_institutions', [])
            for ticker, risk_data in highest_risk[:3]:  # Top 3 riskiest
                if risk_data['contagion_score'] > 0.7:
                    alerts.append({
                        'alert_type': 'HIGH_RISK_INSTITUTION',
                        'severity': 'MEDIUM',
                        'message': f"{ticker} shows high contagion risk ({risk_data['contagion_score']:.3f})",
                        'institution': ticker,
                        'metric_value': risk_data['contagion_score'],
                        'threshold': 0.7,
                        'timestamp': datetime.now()
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error generating contagion alerts: {e}")
            return []


def get_gsib_monitor(config_path: Optional[str] = None) -> GSIBMonitor:
    """Factory function to get G-SIB monitor instance"""
    return GSIBMonitor(config_path)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    monitor = get_gsib_monitor()
    
    print("Testing G-SIB monitoring...")
    
    # Track G-SIB movements
    market_data = monitor.track_global_gsib_movements(period="1mo")
    print(f"Tracked {len(market_data)} G-SIB institutions")
    
    if market_data:
        # Analyze cross-market correlations
        correlation_analysis = monitor.detect_cross_market_correlations(market_data)
        print("Correlation analysis completed")
        
        if correlation_analysis:
            # Calculate systemic risk
            systemic_risk = monitor.calculate_systemic_risk_score(market_data)
            print(f"Systemic risk level: {systemic_risk.get('risk_level', 'UNKNOWN')}")
            print(f"Systemic risk score: {systemic_risk.get('systemic_risk_score', 0):.3f}")
            
            # Generate alerts
            alerts = monitor.generate_contagion_alerts(correlation_analysis)
            print(f"Generated {len(alerts)} contagion alerts")
            
            for alert in alerts:
                print(f"- {alert['alert_type']}: {alert['message']}")
    else:
        print("No market data available for analysis")