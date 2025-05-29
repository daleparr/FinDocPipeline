"""
Market Intelligence Module for BoE Mosaic Lens
Provides Yahoo Finance integration, G-SIB monitoring, and systemic risk analysis
"""

from .yahoo_finance_client import YahooFinanceClient, get_yahoo_finance_client
from .market_indicators import MarketIndicatorsEngine, get_market_indicators_engine
from .sentiment_market_correlator import SentimentMarketCorrelator, get_sentiment_market_correlator
from .gsib_monitor import GSIBMonitor, get_gsib_monitor
from .market_intelligence_dashboard import MarketIntelligenceDashboard, get_market_intelligence_dashboard

__version__ = "1.0.0"
__author__ = "BoE Mosaic Lens Development Team"

__all__ = [
    'YahooFinanceClient',
    'get_yahoo_finance_client',
    'MarketIndicatorsEngine', 
    'get_market_indicators_engine',
    'SentimentMarketCorrelator',
    'get_sentiment_market_correlator',
    'GSIBMonitor',
    'get_gsib_monitor',
    'MarketIntelligenceDashboard',
    'get_market_intelligence_dashboard'
]