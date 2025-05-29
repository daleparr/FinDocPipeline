"""
Yahoo Finance API Client for BoE Mosaic Lens Market Intelligence
Provides real-time and historical market data for G-SIB monitoring
"""

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# Suppress yfinance warnings if available
if YFINANCE_AVAILABLE:
    warnings.filterwarnings("ignore", category=FutureWarning)

class YahooFinanceClient:
    """
    Enhanced Yahoo Finance client for G-SIB market data collection
    with rate limiting, error handling, and BoE-specific features
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Yahoo Finance client with configuration
        
        Args:
            config_path: Path to G-SIB institutions configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Check if yfinance is available
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance library not available. Install with: pip install yfinance")
            raise ImportError("yfinance library is required but not installed")
        
        self.config = self._load_config(config_path)
        self.session = self._setup_session()
        self.rate_limiter = RateLimiter(
            requests_per_hour=self.config.get('rate_limits', {}).get('requests_per_hour', 2000)
        )
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load G-SIB institutions configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "gsib_institutions.yaml"
        
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.get('data_collection', {}).get('yahoo_finance', {}).get('retry_attempts', 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get_gsib_tickers(self, region: Optional[str] = None, bucket: Optional[str] = None) -> List[str]:
        """
        Get list of G-SIB tickers for monitoring
        
        Args:
            region: Filter by region ('us_banks', 'european_banks', 'uk_banks')
            bucket: Filter by FSB bucket ('bucket_1', 'bucket_2', 'bucket_3', 'bucket_4')
            
        Returns:
            List of ticker symbols
        """
        tickers = []
        gsib_data = self.config.get('global_gsibs', {})
        
        if bucket:
            # Filter by FSB systemic importance bucket
            if bucket in gsib_data:
                tickers = [bank['ticker'] for bank in gsib_data[bucket]]
        elif region:
            # Legacy region filtering (for backward compatibility)
            if region in gsib_data:
                tickers = [bank['ticker'] for bank in gsib_data[region]]
        else:
            # Get all tickers from all buckets and regions
            for bucket_or_region in gsib_data.values():
                if isinstance(bucket_or_region, list):
                    tickers.extend([bank['ticker'] for bank in bucket_or_region])
        
        return tickers
    
    def get_all_gsib_tickers(self) -> List[str]:
        """Get complete list of all G-SIB tickers"""
        all_tickers = []
        gsib_data = self.config.get('global_gsibs', {})
        
        # Iterate through all buckets
        for bucket_name, banks in gsib_data.items():
            if isinstance(banks, list):
                all_tickers.extend([bank['ticker'] for bank in banks])
        
        return list(set(all_tickers))  # Remove duplicates
    
    def get_gsib_by_systemic_importance(self) -> Dict[str, List[str]]:
        """Get G-SIBs organized by systemic importance buckets"""
        buckets = {}
        gsib_data = self.config.get('global_gsibs', {})
        
        for bucket_name, banks in gsib_data.items():
            if isinstance(banks, list) and bucket_name.startswith('bucket_'):
                buckets[bucket_name] = [bank['ticker'] for bank in banks]
        
        return buckets
    
    def fetch_market_data(
        self, 
        ticker: str, 
        period: str = "6mo", 
        interval: str = "1d",
        include_dividends: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            include_dividends: Whether to include dividend data
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(
                period=period, 
                interval=interval,
                actions=include_dividends
            )
            
            if hist.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Add calculated fields
            hist = self._add_market_indicators(hist)
            hist['ticker'] = ticker
            
            self.logger.info(f"Fetched {len(hist)} data points for {ticker}")
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_tickers(
        self, 
        tickers: List[str], 
        period: str = "6mo", 
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            self.logger.info(f"Fetching data for {ticker}")
            data = self.fetch_market_data(ticker, period, interval)
            if not data.empty:
                results[ticker] = data
            
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        return results
    
    def get_earnings_calendar(
        self, 
        ticker: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get earnings calendar for a ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for calendar
            end_date: End date for calendar
            
        Returns:
            DataFrame with earnings dates and estimates
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None or calendar.empty:
                self.logger.warning(f"No earnings calendar data for {ticker}")
                return pd.DataFrame()
            
            # Filter by date range if provided
            if start_date or end_date:
                calendar = self._filter_by_date_range(calendar, start_date, end_date)
            
            calendar['ticker'] = ticker
            return calendar
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_options_data(self, ticker: str, expiry_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch options data for volatility analysis
        
        Args:
            ticker: Stock ticker symbol
            expiry_date: Specific expiry date (YYYY-MM-DD format)
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            stock = yf.Ticker(ticker)
            
            if expiry_date:
                options = stock.option_chain(expiry_date)
            else:
                # Get nearest expiry
                expirations = stock.options
                if not expirations:
                    return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
                options = stock.option_chain(expirations[0])
            
            return {
                'calls': options.calls,
                'puts': options.puts
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching options data for {ticker}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def get_institutional_holdings(self, ticker: str) -> pd.DataFrame:
        """
        Get institutional holdings data
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with institutional holdings information
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            stock = yf.Ticker(ticker)
            holders = stock.institutional_holders
            
            if holders is None or holders.empty:
                self.logger.warning(f"No institutional holdings data for {ticker}")
                return pd.DataFrame()
            
            holders['ticker'] = ticker
            return holders
            
        except Exception as e:
            self.logger.error(f"Error fetching institutional holdings for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_volatility(self, ticker: str, window: int = 30) -> pd.DataFrame:
        """
        Calculate historical volatility for a ticker
        
        Args:
            ticker: Stock ticker symbol
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility metrics
        """
        data = self.fetch_market_data(ticker, period="1y", interval="1d")
        
        if data.empty:
            return pd.DataFrame()
        
        # Calculate various volatility measures
        data['returns'] = data['Close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
        data['realized_vol'] = data['returns'].rolling(window=window).std()
        data['vol_of_vol'] = data['volatility'].rolling(window=window).std()
        
        return data[['Close', 'returns', 'volatility', 'realized_vol', 'vol_of_vol']].dropna()
    
    def _add_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to market data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicators
        """
        if df.empty:
            return df
        
        # Daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Volatility (rolling 5-day)
        df['volatility'] = df['daily_return'].rolling(window=5).std()
        
        # Volume ratio (vs 20-day average)
        df['volume_avg_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_avg_20']
        
        # Price momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        # RSI (Relative Strength Index)
        df['rsi'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'], df['bb_middle'] = self._calculate_bollinger_bands(df['Close'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df
    
    def get_gsib_correlation_matrix(self, period: str = "6mo") -> pd.DataFrame:
        """
        Calculate correlation matrix for all G-SIB institutions
        
        Args:
            period: Data period for correlation calculation
            
        Returns:
            Correlation matrix DataFrame
        """
        tickers = self.get_gsib_tickers()
        data_dict = self.fetch_multiple_tickers(tickers, period=period)
        
        # Extract daily returns for correlation
        returns_data = {}
        for ticker, df in data_dict.items():
            if not df.empty and 'daily_return' in df.columns:
                returns_data[ticker] = df['daily_return'].dropna()
        
        if not returns_data:
            return pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def detect_earnings_windows(self, ticker: str, days_before: int = 5, days_after: int = 3) -> List[Dict]:
        """
        Detect earnings announcement windows for analysis
        
        Args:
            ticker: Stock ticker symbol
            days_before: Days before earnings to include
            days_after: Days after earnings to include
            
        Returns:
            List of earnings windows with date ranges
        """
        calendar = self.get_earnings_calendar(ticker)
        
        if calendar.empty:
            return []
        
        windows = []
        for _, row in calendar.iterrows():
            earnings_date = row.name if hasattr(row, 'name') else row.get('Earnings Date')
            
            if earnings_date:
                # Use pandas Timedelta for proper timestamp arithmetic
                earnings_ts = pd.Timestamp(earnings_date)
                window = {
                    'ticker': ticker,
                    'earnings_date': earnings_date,
                    'window_start': earnings_ts - pd.Timedelta(days=days_before),
                    'window_end': earnings_ts + pd.Timedelta(days=days_after),
                    'pre_earnings_start': earnings_ts - pd.Timedelta(days=days_before),
                    'pre_earnings_end': earnings_ts - pd.Timedelta(days=1),
                    'post_earnings_start': earnings_ts + pd.Timedelta(days=1),
                    'post_earnings_end': earnings_ts + pd.Timedelta(days=days_after)
                }
                windows.append(window)
        
        return windows


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, requests_per_hour: int = 2000):
        self.requests_per_hour = requests_per_hour
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove requests older than 1 hour
        self.requests = [req_time for req_time in self.requests if now - req_time < 3600]
        
        # Check if we need to wait
        if len(self.requests) >= self.requests_per_hour:
            sleep_time = 3600 - (now - self.requests[0]) + 1
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.requests = []
        
        # Record this request
        self.requests.append(now)


# Utility functions for market intelligence
def get_yahoo_finance_client(config_path: Optional[str] = None) -> YahooFinanceClient:
    """Factory function to get Yahoo Finance client instance"""
    return YahooFinanceClient(config_path)


def fetch_gsib_market_snapshot(client: Optional[YahooFinanceClient] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch current market snapshot for all G-SIB institutions
    
    Args:
        client: Yahoo Finance client instance
        
    Returns:
        Dictionary mapping ticker to current market data
    """
    if client is None:
        client = get_yahoo_finance_client()
    
    tickers = client.get_gsib_tickers()
    return client.fetch_multiple_tickers(tickers, period="1d", interval="1m")


def calculate_systemic_risk_indicators(market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate systemic risk indicators from market data
    
    Args:
        market_data: Dictionary mapping ticker to market data
        
    Returns:
        Dictionary of systemic risk metrics
    """
    if not market_data:
        return {}
    
    # Extract latest returns
    latest_returns = {}
    for ticker, df in market_data.items():
        if not df.empty and 'daily_return' in df.columns:
            latest_return = df['daily_return'].iloc[-1]
            if not pd.isna(latest_return):
                latest_returns[ticker] = latest_return
    
    if not latest_returns:
        return {}
    
    returns_array = np.array(list(latest_returns.values()))
    
    # Calculate systemic risk metrics
    metrics = {
        'average_return': np.mean(returns_array),
        'return_volatility': np.std(returns_array),
        'max_drawdown': np.min(returns_array),
        'correlation_stress': len([r for r in returns_array if r < -0.02]) / len(returns_array),  # % with >2% decline
        'systemic_risk_score': np.std(returns_array) * np.abs(np.mean(returns_array)) if np.mean(returns_array) < 0 else 0
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    client = get_yahoo_finance_client()
    
    # Test fetching data for a single ticker
    print("Testing single ticker fetch...")
    data = client.fetch_market_data("JPM", period="1mo", interval="1d")
    print(f"Fetched {len(data)} data points for JPM")
    print(data.head())
    
    # Test G-SIB correlation matrix
    print("\nTesting G-SIB correlation matrix...")
    correlation_matrix = client.get_gsib_correlation_matrix(period="3mo")
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(correlation_matrix.head())
    
    # Test systemic risk indicators
    print("\nTesting systemic risk indicators...")
    snapshot = fetch_gsib_market_snapshot(client)
    risk_metrics = calculate_systemic_risk_indicators(snapshot)
    print("Systemic risk metrics:", risk_metrics)