"""
Market Indicators Engine for BoE Mosaic Lens
Calculates advanced market indicators and risk metrics for G-SIB monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class MarketIndicatorsEngine:
    """
    Advanced market indicators calculation engine for G-SIB risk assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market indicators engine
        
        Args:
            config: Configuration dictionary with thresholds and parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for market indicators"""
        return {
            'volatility_windows': [5, 10, 20, 30],
            'momentum_windows': [5, 10, 20],
            'volume_windows': [10, 20, 50],
            'correlation_window': 30,
            'outlier_threshold': 3.0,
            'liquidity_threshold': 0.02,
            'stress_threshold': 0.05
        }
    
    def calculate_comprehensive_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive market indicators for a single ticker
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        if market_data.empty:
            return market_data
        
        df = market_data.copy()
        
        # Basic price indicators
        df = self._add_price_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Liquidity indicators
        df = self._add_liquidity_indicators(df)
        
        # Risk indicators
        df = self._add_risk_indicators(df)
        
        # Market microstructure indicators
        df = self._add_microstructure_indicators(df)
        
        return df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based indicators"""
        # Daily returns
        df['daily_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price gaps
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_up'] = (df['gap'] > 0.01).astype(int)
        df['gap_down'] = (df['gap'] < -0.01).astype(int)
        
        # Intraday range
        df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
        df['true_range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        for window in self.config['volatility_windows']:
            # Rolling volatility
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std()
            
            # Realized volatility (annualized)
            df[f'realized_vol_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            
            # Volatility of volatility
            df[f'vol_of_vol_{window}d'] = df[f'volatility_{window}d'].rolling(window=window).std()
            
            # GARCH-like volatility clustering
            df[f'vol_clustering_{window}d'] = (
                df[f'volatility_{window}d'] > df[f'volatility_{window}d'].rolling(window=window*2).mean()
            ).astype(int)
        
        # Parkinson volatility (using high-low range)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['High'] / df['Low']) ** 2
        ).rolling(window=20).mean()
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['High'] / df['Low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2
        ).rolling(window=20).mean()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        for window in self.config['volume_windows']:
            # Volume moving averages
            df[f'volume_ma_{window}d'] = df['Volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}d'] = df['Volume'] / df[f'volume_ma_{window}d']
            
            # Volume-weighted average price
            df[f'vwap_{window}d'] = (
                (df['Close'] * df['Volume']).rolling(window=window).sum() /
                df['Volume'].rolling(window=window).sum()
            )
            
            # Volume rate of change
            df[f'volume_roc_{window}d'] = df['Volume'].pct_change(periods=window)
        
        # On-balance volume
        df['obv'] = (df['Volume'] * np.sign(df['daily_return'])).cumsum()
        
        # Accumulation/Distribution line
        df['ad_line'] = (
            ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / 
            (df['High'] - df['Low']) * df['Volume']
        ).cumsum()
        
        # Volume-price trend
        df['vpt'] = (df['daily_return'] * df['Volume']).cumsum()
        
        # Ease of movement
        df['eom'] = (
            (df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2
        ) * df['Volume'] / (df['High'] - df['Low'])
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        for window in self.config['momentum_windows']:
            # Price momentum
            df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
            
            # Rate of change
            df[f'roc_{window}d'] = df['Close'].pct_change(periods=window)
            
            # Relative strength
            df[f'rs_{window}d'] = df['Close'] / df['Close'].rolling(window=window).mean()
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_30'] = self._calculate_rsi(df['Close'], 30)
        
        # Stochastic oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['Close'])
        
        # Commodity Channel Index
        df['cci'] = self._calculate_cci(df)
        
        return df
    
    def _add_liquidity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-based indicators"""
        # Bid-ask spread proxy (using high-low range)
        df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Amihud illiquidity ratio
        df['amihud_illiq'] = np.abs(df['daily_return']) / (df['Volume'] * df['Close'])
        
        # Turnover ratio
        df['turnover'] = df['Volume'] / df['Volume'].rolling(window=252).mean()  # vs annual average
        
        # Price impact
        df['price_impact'] = np.abs(df['daily_return']) / np.log(df['Volume'] + 1)
        
        # Liquidity stress indicator
        df['liquidity_stress'] = (
            (df['spread_proxy'] > df['spread_proxy'].rolling(window=30).quantile(0.9)) |
            (df['amihud_illiq'] > df['amihud_illiq'].rolling(window=30).quantile(0.9))
        ).astype(int)
        
        return df
    
    def _add_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-based indicators"""
        # Value at Risk (VaR) - 5% and 1%
        df['var_5pct'] = df['daily_return'].rolling(window=252).quantile(0.05)
        df['var_1pct'] = df['daily_return'].rolling(window=252).quantile(0.01)
        
        # Expected Shortfall (Conditional VaR)
        df['es_5pct'] = df['daily_return'].rolling(window=252).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        
        # Maximum drawdown
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()
        df['running_max'] = df['cumulative_return'].expanding().max()
        df['drawdown'] = (df['cumulative_return'] - df['running_max']) / df['running_max']
        df['max_drawdown'] = df['drawdown'].expanding().min()
        
        # Downside deviation
        df['downside_dev'] = df['daily_return'].rolling(window=30).apply(
            lambda x: np.sqrt(np.mean(np.minimum(x, 0) ** 2))
        )
        
        # Skewness and kurtosis
        df['skewness'] = df['daily_return'].rolling(window=30).skew()
        df['kurtosis'] = df['daily_return'].rolling(window=30).kurt()
        
        # Tail risk indicator
        df['tail_risk'] = (
            (df['skewness'] < -1) | (df['kurtosis'] > 5)
        ).astype(int)
        
        return df
    
    def _add_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure indicators"""
        # Price efficiency measures
        df['price_delay'] = self._calculate_price_delay(df['daily_return'])
        
        # Autocorrelation of returns
        df['return_autocorr'] = df['daily_return'].rolling(window=30).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        )
        
        # Variance ratio test statistic
        df['variance_ratio'] = self._calculate_variance_ratio(df['daily_return'])
        
        # Market efficiency score
        df['efficiency_score'] = (
            (np.abs(df['return_autocorr']) < 0.1) & 
            (np.abs(df['variance_ratio'] - 1) < 0.2)
        ).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_window).min()
        highest_high = df['High'].rolling(window=k_window).max()
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        return -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_cci(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)
    
    def _calculate_price_delay(self, returns: pd.Series, lags: int = 5) -> pd.Series:
        """Calculate price delay measure"""
        def delay_measure(x):
            if len(x) < lags + 1:
                return np.nan
            
            # Regression of current return on lagged returns
            y = x[lags:].values
            X = np.column_stack([x[i:-(lags-i)].values for i in range(lags)])
            
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                return 1 - model.score(X, y)  # 1 - R²
            except:
                return np.nan
        
        return returns.rolling(window=30).apply(delay_measure)
    
    def _calculate_variance_ratio(self, returns: pd.Series, q: int = 2) -> pd.Series:
        """Calculate variance ratio test statistic"""
        def vr_statistic(x):
            if len(x) < q * 2:
                return np.nan
            
            # Variance of q-period returns
            q_returns = x.rolling(window=q).sum()[q-1::q]
            var_q = q_returns.var()
            
            # Variance of 1-period returns
            var_1 = x.var()
            
            # Variance ratio
            return var_q / (q * var_1) if var_1 != 0 else np.nan
        
        return returns.rolling(window=60).apply(vr_statistic)
    
    def calculate_cross_sectional_indicators(self, market_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate cross-sectional indicators across multiple tickers
        
        Args:
            market_data_dict: Dictionary mapping ticker to market data
            
        Returns:
            DataFrame with cross-sectional indicators
        """
        if not market_data_dict:
            return pd.DataFrame()
        
        # Extract returns for all tickers
        returns_dict = {}
        for ticker, df in market_data_dict.items():
            if not df.empty and 'daily_return' in df.columns:
                returns_dict[ticker] = df['daily_return']
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate cross-sectional indicators
        indicators = pd.DataFrame(index=returns_df.index)
        
        # Cross-sectional dispersion
        indicators['cross_sectional_vol'] = returns_df.std(axis=1)
        indicators['cross_sectional_range'] = returns_df.max(axis=1) - returns_df.min(axis=1)
        
        # Market beta (vs equal-weighted portfolio)
        market_return = returns_df.mean(axis=1)
        for ticker in returns_df.columns:
            indicators[f'{ticker}_beta'] = returns_df[ticker].rolling(window=60).cov(market_return) / market_return.rolling(window=60).var()
        
        # Correlation indicators
        indicators['avg_correlation'] = returns_df.rolling(window=30).corr().groupby(level=0).mean().mean(axis=1)
        indicators['max_correlation'] = returns_df.rolling(window=30).corr().groupby(level=0).max().max(axis=1)
        indicators['min_correlation'] = returns_df.rolling(window=30).corr().groupby(level=0).min().min(axis=1)
        
        # Systemic risk indicators
        indicators['systemic_stress'] = (returns_df < returns_df.quantile(0.05, axis=1).values.reshape(-1, 1)).sum(axis=1) / len(returns_df.columns)
        indicators['contagion_risk'] = (indicators['avg_correlation'] > 0.7).astype(int)
        
        return indicators
    
    def detect_market_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market anomalies and unusual patterns
        
        Args:
            df: DataFrame with market indicators
            
        Returns:
            DataFrame with anomaly flags
        """
        anomalies = pd.DataFrame(index=df.index)
        
        # Price anomalies
        if 'daily_return' in df.columns:
            return_zscore = np.abs(stats.zscore(df['daily_return'].dropna()))
            anomalies['price_anomaly'] = (return_zscore > self.config['outlier_threshold']).astype(int)
        
        # Volume anomalies
        if 'volume_ratio_20d' in df.columns:
            anomalies['volume_anomaly'] = (df['volume_ratio_20d'] > 3.0).astype(int)
        
        # Volatility anomalies
        if 'volatility_20d' in df.columns:
            vol_zscore = np.abs(stats.zscore(df['volatility_20d'].dropna()))
            anomalies['volatility_anomaly'] = (vol_zscore > 2.0).astype(int)
        
        # Liquidity anomalies
        if 'liquidity_stress' in df.columns:
            anomalies['liquidity_anomaly'] = df['liquidity_stress']
        
        # Gap anomalies
        if 'gap' in df.columns:
            anomalies['gap_anomaly'] = (np.abs(df['gap']) > 0.05).astype(int)  # >5% gap
        
        # Combined anomaly score
        anomaly_cols = [col for col in anomalies.columns if col.endswith('_anomaly')]
        if anomaly_cols:
            anomalies['total_anomaly_score'] = anomalies[anomaly_cols].sum(axis=1)
            anomalies['high_anomaly_flag'] = (anomalies['total_anomaly_score'] >= 3).astype(int)
        
        return anomalies
    
    def calculate_earnings_impact_metrics(self, df: pd.DataFrame, earnings_date: datetime) -> Dict[str, float]:
        """
        Calculate earnings announcement impact metrics
        
        Args:
            df: DataFrame with market data
            earnings_date: Date of earnings announcement
            
        Returns:
            Dictionary with earnings impact metrics
        """
        # Define pre and post earnings windows - use pandas Timedelta
        earnings_ts = pd.Timestamp(earnings_date)
        pre_start = earnings_ts - pd.Timedelta(days=5)
        pre_end = earnings_ts - pd.Timedelta(days=1)
        post_start = earnings_ts + pd.Timedelta(days=1)
        post_end = earnings_ts + pd.Timedelta(days=3)
        
        # Filter data for each window
        pre_data = df[(df.index >= pre_start) & (df.index <= pre_end)]
        post_data = df[(df.index >= post_start) & (df.index <= post_end)]
        
        if pre_data.empty or post_data.empty:
            return {}
        
        metrics = {}
        
        # Return metrics
        if 'daily_return' in df.columns:
            metrics['pre_earnings_return'] = pre_data['daily_return'].sum()
            metrics['post_earnings_return'] = post_data['daily_return'].sum()
            metrics['earnings_surprise'] = post_data['daily_return'].iloc[0] if len(post_data) > 0 else 0
        
        # Volatility metrics
        if 'volatility_5d' in df.columns:
            metrics['pre_earnings_vol'] = pre_data['volatility_5d'].mean()
            metrics['post_earnings_vol'] = post_data['volatility_5d'].mean()
            metrics['vol_change'] = metrics['post_earnings_vol'] - metrics['pre_earnings_vol']
        
        # Volume metrics
        if 'volume_ratio_20d' in df.columns:
            metrics['pre_earnings_volume'] = pre_data['volume_ratio_20d'].mean()
            metrics['post_earnings_volume'] = post_data['volume_ratio_20d'].mean()
            metrics['volume_spike'] = max(post_data['volume_ratio_20d'].max(), pre_data['volume_ratio_20d'].max())
        
        return metrics


def get_market_indicators_engine(config: Optional[Dict] = None) -> MarketIndicatorsEngine:
    """Factory function to get market indicators engine instance"""
    return MarketIndicatorsEngine(config)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from yahoo_finance_client import get_yahoo_finance_client
    
    # Initialize components
    client = get_yahoo_finance_client()
    engine = get_market_indicators_engine()
    
    # Test with sample data
    print("Testing market indicators engine...")
    
    # Fetch sample data
    data = client.fetch_market_data("JPM", period="3mo", interval="1d")
    
    if not data.empty:
        # Calculate comprehensive indicators
        enhanced_data = engine.calculate_comprehensive_indicators(data)
        print(f"Added {len(enhanced_data.columns) - len(data.columns)} new indicators")
        print("New columns:", [col for col in enhanced_data.columns if col not in data.columns][:10])
        
        # Detect anomalies
        anomalies = engine.detect_market_anomalies(enhanced_data)
        print(f"Detected {anomalies['total_anomaly_score'].sum()} total anomalies")
        
        # Test cross-sectional indicators
        tickers = ["JPM", "BAC", "C"]
        multi_data = client.fetch_multiple_tickers(tickers, period="1mo")
        
        if multi_data:
            cross_indicators = engine.calculate_cross_sectional_indicators(multi_data)
            print(f"Cross-sectional indicators shape: {cross_indicators.shape}")
    else:
        print("No data available for testing")