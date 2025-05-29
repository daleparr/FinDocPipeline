"""
Sentiment-Market Correlator for BoE Mosaic Lens
Implements the core functionality to correlate NLP sentiment with market movements
Based on the provided pseudocode for hybrid risk detection
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
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class SentimentMarketCorrelator:
    """
    Core engine for correlating NLP sentiment signals with market movements
    Implements hybrid risk detection combining sentiment and market data
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize sentiment-market correlator
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "gsib_institutions.yaml"
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('market_intelligence_config', {})
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'correlation_thresholds': {
                'low_correlation': 0.3,
                'medium_correlation': 0.5,
                'high_correlation': 0.7,
                'systemic_risk': 0.8
            },
            'earnings_windows': {
                'pre_earnings_days': 5,
                'post_earnings_days': 3,
                'extended_analysis_days': 14
            },
            'alert_severity_mapping': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'critical': 0.9
            }
        }
    
    def fetch_nlp_signals(self, bank: str, quarter: str, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load NLP-generated signals for given bank & quarter.
        Returns a DataFrame with columns:
          - date (date of call)
          - topic_label (e.g. 'Loan Performance')
          - sentiment_score (float, negative = risk)
        
        Args:
            bank: Bank name
            quarter: Quarter identifier (e.g., 'Q1_2025')
            data_path: Optional path to NLP results
            
        Returns:
            DataFrame with NLP signals
        """
        try:
            # Try to load from processed data directory
            if data_path is None:
                data_path = Path(__file__).parent.parent.parent / "data" / "metadata" / "versions"
            
            # Look for processed NLP data
            nlp_files = []
            for bank_dir in Path(data_path).glob(f"*{bank}*"):
                for quarter_dir in bank_dir.glob(f"*{quarter}*"):
                    for version_dir in quarter_dir.iterdir():
                        if version_dir.is_dir():
                            parquet_file = version_dir / "processed_data.parquet"
                            if parquet_file.exists():
                                nlp_files.append(parquet_file)
            
            if not nlp_files:
                self.logger.warning(f"No NLP data found for {bank} {quarter}")
                return self._generate_sample_nlp_data(bank, quarter)
            
            # Load the most recent file
            latest_file = max(nlp_files, key=lambda x: x.stat().st_mtime)
            nlp_df = pd.read_parquet(latest_file)
            
            # Transform to expected format
            nlp_signals = self._transform_nlp_data(nlp_df, bank, quarter)
            
            self.logger.info(f"Loaded {len(nlp_signals)} NLP signals for {bank} {quarter}")
            return nlp_signals
            
        except Exception as e:
            self.logger.error(f"Error loading NLP signals for {bank} {quarter}: {e}")
            return self._generate_sample_nlp_data(bank, quarter)
    
    def _transform_nlp_data(self, nlp_df: pd.DataFrame, bank: str, quarter: str) -> pd.DataFrame:
        """Transform raw NLP data to expected format"""
        try:
            # Extract relevant columns and create expected format
            signals = []
            
            self.logger.info(f"Transforming NLP data with columns: {nlp_df.columns.tolist()}")
            
            # Check for actual data structure
            if 'timestamp' in nlp_df.columns and 'topic_label' in nlp_df.columns:
                # Use actual data structure
                for _, row in nlp_df.iterrows():
                    # Parse timestamp
                    try:
                        if pd.isna(row['timestamp']) or row['timestamp'] == 'unknown':
                            # Use processing date as fallback
                            date = pd.to_datetime(row.get('processing_date', datetime.now()))
                        else:
                            date = pd.to_datetime(row['timestamp'])
                    except:
                        date = pd.to_datetime(row.get('processing_date', datetime.now()))
                    
                    # Calculate sentiment score from topic confidence (proxy)
                    sentiment_score = row.get('topic_confidence', 0.5) - 0.5  # Convert 0-1 to -0.5 to 0.5
                    
                    signals.append({
                        'date': date,
                        'topic_label': row['topic_label'],
                        'sentiment_score': sentiment_score,
                        'topic_keywords': row.get('topic_keywords', ''),
                        'text': row.get('text', ''),
                        'bank': bank,
                        'quarter': quarter
                    })
            
            elif 'date' in nlp_df.columns and 'topic' in nlp_df.columns:
                # Legacy format
                grouped = nlp_df.groupby(['date', 'topic'])
                for (date, topic), group in grouped:
                    sentiment_score = group['sentiment_compound'].mean() if 'sentiment_compound' in group.columns else 0
                    signals.append({
                        'date': pd.to_datetime(date),
                        'topic_label': topic,
                        'sentiment_score': sentiment_score,
                        'bank': bank,
                        'quarter': quarter
                    })
            else:
                # Fallback: create aggregated signals from available data
                self.logger.warning(f"Using fallback transformation for {bank} {quarter}")
                
                # Use processing date as base
                base_date = pd.Timestamp.now() - pd.Timedelta(days=30)  # Approximate earnings date
                if 'processing_date' in nlp_df.columns and not nlp_df['processing_date'].empty:
                    try:
                        base_date = pd.to_datetime(nlp_df['processing_date'].iloc[0])
                    except:
                        pass
                
                # Extract topics from data if available
                if 'topic_label' in nlp_df.columns:
                    topics = nlp_df['topic_label'].unique().tolist()
                else:
                    topics = ['Credit Risk', 'Operational Risk', 'Market Risk', 'Regulatory Compliance', 'Financial Performance']
                
                for i, topic in enumerate(topics):
                    # Use topic confidence as sentiment proxy
                    if 'topic_confidence' in nlp_df.columns:
                        topic_data = nlp_df[nlp_df['topic_label'] == topic] if 'topic_label' in nlp_df.columns else nlp_df
                        avg_confidence = topic_data['topic_confidence'].mean() if not topic_data.empty else 0.5
                        sentiment_score = avg_confidence - 0.5  # Convert to sentiment range
                    else:
                        sentiment_score = np.random.normal(-0.1, 0.3)  # Slightly negative bias
                    
                    signals.append({
                        'date': base_date + timedelta(days=i),
                        'topic_label': topic,
                        'sentiment_score': sentiment_score,
                        'bank': bank,
                        'quarter': quarter
                    })
            
            result_df = pd.DataFrame(signals)
            self.logger.info(f"Transformed to {len(result_df)} NLP signals")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error transforming NLP data: {e}")
            return self._generate_sample_nlp_data(bank, quarter)
    
    def _generate_sample_nlp_data(self, bank: str, quarter: str) -> pd.DataFrame:
        """Generate sample NLP data for testing"""
        np.random.seed(42)  # For reproducible results
        
        base_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        topics = ['Credit Risk', 'Operational Risk', 'Market Risk', 'Regulatory Compliance', 'Financial Performance']
        
        signals = []
        for i in range(10):  # 10 days of data
            for topic in topics:
                signals.append({
                    'date': base_date + timedelta(days=i),
                    'topic_label': topic,
                    'sentiment_score': np.random.normal(-0.1, 0.3),  # Slightly negative bias
                    'bank': bank,
                    'quarter': quarter
                })
        
        return pd.DataFrame(signals)
    
    def fetch_market_data(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Use yfinance to fetch historic stock data.
        Returns a DataFrame indexed by date with columns:
          - Close (closing price)
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with market data
        """
        try:
            # Import here to avoid circular imports
            from .yahoo_finance_client import get_yahoo_finance_client
            
            client = get_yahoo_finance_client()
            hist = client.fetch_market_data(ticker, period, interval)
            
            if hist.empty:
                self.logger.warning(f"No market data returned for {ticker}")
                return pd.DataFrame()
            
            # Ensure we have the required columns
            required_cols = ['Close']
            if not all(col in hist.columns for col in required_cols):
                self.logger.error(f"Missing required columns in market data for {ticker}")
                return pd.DataFrame()
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {ticker}: {e}")
            return pd.DataFrame()
    
    def compute_market_indicators(self, hist_df: pd.DataFrame) -> pd.DataFrame:
        """
        From raw price data, compute daily return and volatility.
        Adds columns:
          - daily_return = (Close_today - Close_yesterday) / Close_yesterday
          - volatility   = rolling_std(daily_return, window=5)
        
        Args:
            hist_df: DataFrame with market data
            
        Returns:
            DataFrame with additional indicators
        """
        if hist_df.empty:
            return hist_df
        
        df = hist_df.copy()
        
        # Daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Rolling volatility
        df['volatility'] = df['daily_return'].rolling(window=5).std()
        
        # Additional useful indicators
        df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std()
        
        # Volume-based indicators if available
        if 'Volume' in df.columns:
            df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # Price momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        
        return df
    
    def merge_nlp_and_market(self, nlp_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align NLP signals with market indicators by date.
        Returns a merged DataFrame with columns:
          - date, topic_label, sentiment_score
          - daily_return, volatility
        
        Args:
            nlp_df: DataFrame with NLP signals
            market_df: DataFrame with market data
            
        Returns:
            Merged DataFrame
        """
        if nlp_df.empty or market_df.empty:
            self.logger.warning("Empty input data for merge operation")
            return pd.DataFrame()
        
        try:
            # Ensure both DataFrames use the same date format and index
            market_daily = market_df[['daily_return', 'volatility']].copy()
            
            # Reset index to make date a column for merging
            if isinstance(market_daily.index, pd.DatetimeIndex):
                market_daily = market_daily.reset_index()
                market_daily.rename(columns={'Date': 'date'}, inplace=True)
            
            # Ensure date columns are datetime
            nlp_df['date'] = pd.to_datetime(nlp_df['date'])
            market_daily['date'] = pd.to_datetime(market_daily['date'])
            
            # Merge on date
            merged = nlp_df.merge(
                market_daily,
                on='date',
                how='left'
            )
            
            # Forward fill missing market data (for weekends/holidays)
            merged['daily_return'] = merged['daily_return'].fillna(method='ffill')
            merged['volatility'] = merged['volatility'].fillna(method='ffill')
            
            self.logger.info(f"Merged data shape: {merged.shape}")
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging NLP and market data: {e}")
            return pd.DataFrame()
    
    def flag_combined_alerts(
        self, 
        merged_df: pd.DataFrame, 
        sentiment_thresh: float = -0.3, 
        return_thresh: float = 0.02
    ) -> pd.DataFrame:
        """
        Create boolean flags:
          - market_flag = abs(daily_return) > return_thresh
          - nlp_flag    = sentiment_score < sentiment_thresh
          - risk_alert  = market_flag AND nlp_flag
        
        Args:
            merged_df: Merged DataFrame with NLP and market data
            sentiment_thresh: Threshold for negative sentiment
            return_thresh: Threshold for significant market movement
            
        Returns:
            DataFrame with alert flags
        """
        if merged_df.empty:
            return merged_df
        
        df = merged_df.copy()
        
        # Market movement flag
        df['market_flag'] = df['daily_return'].abs() > return_thresh
        
        # NLP sentiment flag
        df['nlp_flag'] = df['sentiment_score'] < sentiment_thresh
        
        # Combined risk alert
        df['risk_alert'] = df['market_flag'] & df['nlp_flag']
        
        # Additional alert types
        df['divergence_alert'] = (
            (df['sentiment_score'] < -0.2) & (df['daily_return'] > 0.01)  # Negative sentiment, positive return
        ) | (
            (df['sentiment_score'] > 0.2) & (df['daily_return'] < -0.01)  # Positive sentiment, negative return
        )
        
        # Volatility alert
        if 'volatility' in df.columns:
            vol_threshold = df['volatility'].quantile(0.9) if not df['volatility'].isna().all() else 0.02
            df['volatility_alert'] = df['volatility'] > vol_threshold
        
        # Combined high-risk alert
        alert_cols = ['risk_alert', 'divergence_alert']
        if 'volatility_alert' in df.columns:
            alert_cols.append('volatility_alert')
        
        df['high_risk_alert'] = df[alert_cols].any(axis=1)
        
        return df
    
    def run_hybrid_risk_detection(
        self, 
        bank: str, 
        ticker: str, 
        quarter: str,
        sentiment_thresh: float = -0.3,
        return_thresh: float = 0.02
    ) -> pd.DataFrame:
        """
        End-to-end orchestrator:
          1. Fetch NLP signals
          2. Fetch market data
          3. Compute market indicators
          4. Merge datasets
          5. Flag combined alerts
          6. Return alerts for supervisor review
        
        Args:
            bank: Bank name
            ticker: Stock ticker
            quarter: Quarter identifier
            sentiment_thresh: Sentiment threshold for alerts
            return_thresh: Return threshold for alerts
            
        Returns:
            DataFrame with risk alerts
        """
        try:
            self.logger.info(f"Starting hybrid risk detection for {bank} ({ticker}) {quarter}")
            
            # 1. NLP signals
            nlp_df = self.fetch_nlp_signals(bank, quarter)
            if nlp_df.empty:
                self.logger.warning("No NLP signals available")
                return pd.DataFrame()
            
            # 2. Market data
            market_raw = self.fetch_market_data(ticker)
            if market_raw.empty:
                self.logger.warning("No market data available")
                return pd.DataFrame()
            
            # 3. Indicators
            market_indicators = self.compute_market_indicators(market_raw)
            
            # 4. Merge
            combined = self.merge_nlp_and_market(nlp_df, market_indicators)
            if combined.empty:
                self.logger.warning("Failed to merge NLP and market data")
                return pd.DataFrame()
            
            # 5. Flag alerts
            results = self.flag_combined_alerts(combined, sentiment_thresh, return_thresh)
            
            # 6. Extract final alerts
            alerts = results[results['risk_alert'] | results['high_risk_alert']]
            
            self.logger.info(f"Generated {len(alerts)} risk alerts")
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error in hybrid risk detection: {e}")
            return pd.DataFrame()
    
    def calculate_correlation_metrics(self, merged_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate correlation metrics between sentiment and market data
        
        Args:
            merged_df: Merged DataFrame with NLP and market data
            
        Returns:
            Dictionary with correlation metrics
        """
        if merged_df.empty or 'sentiment_score' not in merged_df.columns or 'daily_return' not in merged_df.columns:
            return {}
        
        # Remove NaN values
        clean_data = merged_df[['sentiment_score', 'daily_return']].dropna()
        
        if len(clean_data) < 10:  # Need minimum data points
            return {}
        
        try:
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(clean_data['sentiment_score'], clean_data['daily_return'])
            
            # Spearman correlation (rank-based)
            spearman_corr, spearman_p = stats.spearmanr(clean_data['sentiment_score'], clean_data['daily_return'])
            
            # Kendall's tau
            kendall_corr, kendall_p = stats.kendalltau(clean_data['sentiment_score'], clean_data['daily_return'])
            
            # Linear regression R²
            X = clean_data['sentiment_score'].values.reshape(-1, 1)
            y = clean_data['daily_return'].values
            reg = LinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
            
            return {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'kendall_correlation': kendall_corr,
                'kendall_p_value': kendall_p,
                'r_squared': r_squared,
                'sample_size': len(clean_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation metrics: {e}")
            return {}
    
    def analyze_earnings_impact(
        self, 
        merged_df: pd.DataFrame, 
        earnings_date: datetime,
        pre_days: int = 5,
        post_days: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze the impact of earnings announcements on sentiment-market correlation
        
        Args:
            merged_df: Merged DataFrame with NLP and market data
            earnings_date: Date of earnings announcement
            pre_days: Days before earnings to analyze
            post_days: Days after earnings to analyze
            
        Returns:
            Dictionary with earnings impact analysis
        """
        if merged_df.empty:
            return {}
        
        try:
            # Define windows - use pandas Timedelta for proper timestamp arithmetic
            earnings_ts = pd.Timestamp(earnings_date)
            pre_start = earnings_ts - pd.Timedelta(days=pre_days)
            pre_end = earnings_ts - pd.Timedelta(days=1)
            post_start = earnings_ts + pd.Timedelta(days=1)
            post_end = earnings_ts + pd.Timedelta(days=post_days)
            
            # Filter data
            pre_data = merged_df[(merged_df['date'] >= pre_start) & (merged_df['date'] <= pre_end)]
            post_data = merged_df[(merged_df['date'] >= post_start) & (merged_df['date'] <= post_end)]
            
            analysis = {
                'earnings_date': earnings_date,
                'pre_earnings_period': f"{pre_start.date()} to {pre_end.date()}",
                'post_earnings_period': f"{post_start.date()} to {post_end.date()}"
            }
            
            # Pre-earnings analysis
            if not pre_data.empty:
                analysis['pre_earnings'] = {
                    'avg_sentiment': pre_data['sentiment_score'].mean(),
                    'avg_return': pre_data['daily_return'].mean(),
                    'correlation': self.calculate_correlation_metrics(pre_data),
                    'alert_count': pre_data['risk_alert'].sum() if 'risk_alert' in pre_data.columns else 0
                }
            
            # Post-earnings analysis
            if not post_data.empty:
                analysis['post_earnings'] = {
                    'avg_sentiment': post_data['sentiment_score'].mean(),
                    'avg_return': post_data['daily_return'].mean(),
                    'correlation': self.calculate_correlation_metrics(post_data),
                    'alert_count': post_data['risk_alert'].sum() if 'risk_alert' in post_data.columns else 0
                }
            
            # Compare pre vs post
            if 'pre_earnings' in analysis and 'post_earnings' in analysis:
                analysis['comparison'] = {
                    'sentiment_change': analysis['post_earnings']['avg_sentiment'] - analysis['pre_earnings']['avg_sentiment'],
                    'return_change': analysis['post_earnings']['avg_return'] - analysis['pre_earnings']['avg_return'],
                    'correlation_change': (
                        analysis['post_earnings']['correlation'].get('pearson_correlation', 0) - 
                        analysis['pre_earnings']['correlation'].get('pearson_correlation', 0)
                    )
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing earnings impact: {e}")
            return {}
    
    def generate_risk_summary(self, alerts_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of risk alerts for supervisor review
        
        Args:
            alerts_df: DataFrame with risk alerts
            
        Returns:
            Dictionary with risk summary
        """
        if alerts_df.empty:
            return {'total_alerts': 0, 'risk_level': 'LOW'}
        
        summary = {
            'total_alerts': len(alerts_df),
            'alert_breakdown': {},
            'risk_level': 'LOW',
            'key_concerns': [],
            'recommendations': []
        }
        
        # Count different types of alerts
        alert_types = ['risk_alert', 'divergence_alert', 'volatility_alert', 'high_risk_alert']
        for alert_type in alert_types:
            if alert_type in alerts_df.columns:
                summary['alert_breakdown'][alert_type] = alerts_df[alert_type].sum()
        
        # Determine overall risk level
        high_risk_count = summary['alert_breakdown'].get('high_risk_alert', 0)
        total_alerts = summary['total_alerts']
        
        if high_risk_count > 5 or total_alerts > 20:
            summary['risk_level'] = 'CRITICAL'
        elif high_risk_count > 2 or total_alerts > 10:
            summary['risk_level'] = 'HIGH'
        elif total_alerts > 5:
            summary['risk_level'] = 'MEDIUM'
        
        # Generate key concerns
        if 'sentiment_score' in alerts_df.columns:
            avg_sentiment = alerts_df['sentiment_score'].mean()
            if avg_sentiment < -0.5:
                summary['key_concerns'].append("Persistently negative sentiment across multiple topics")
        
        if 'daily_return' in alerts_df.columns:
            volatile_days = (alerts_df['daily_return'].abs() > 0.03).sum()
            if volatile_days > 3:
                summary['key_concerns'].append(f"High market volatility on {volatile_days} days")
        
        # Generate recommendations
        if summary['risk_level'] in ['HIGH', 'CRITICAL']:
            summary['recommendations'].append("Immediate supervisor review recommended")
            summary['recommendations'].append("Consider enhanced monitoring of institution")
        
        if summary['alert_breakdown'].get('divergence_alert', 0) > 3:
            summary['recommendations'].append("Investigate disconnect between management narrative and market reaction")
        
        return summary


def get_sentiment_market_correlator(config_path: Optional[str] = None) -> SentimentMarketCorrelator:
    """Factory function to get sentiment-market correlator instance"""
    return SentimentMarketCorrelator(config_path)


if __name__ == "__main__":
    # Example usage matching the provided pseudocode
    logging.basicConfig(level=logging.INFO)
    
    correlator = get_sentiment_market_correlator()
    
    # Example usage as per pseudocode
    print("Testing hybrid risk detection...")
    alerts = correlator.run_hybrid_risk_detection(
        bank="JPMorganChase", 
        ticker="JPM", 
        quarter="Q1_2025"
    )
    
    if not alerts.empty:
        print(f"Generated {len(alerts)} alerts")
        print("\nAlert summary:")
        print(alerts[['date', 'topic_label', 'sentiment_score', 'daily_return', 'risk_alert']].head())
        
        # Generate risk summary
        risk_summary = correlator.generate_risk_summary(alerts)
        print(f"\nRisk Level: {risk_summary['risk_level']}")
        print(f"Total Alerts: {risk_summary['total_alerts']}")
    else:
        print("No alerts generated")