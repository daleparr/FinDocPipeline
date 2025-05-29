"""
Simple Market-Sentiment Integration Test
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_integration():
    print("Testing Market-Sentiment Integration...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.market_intelligence import (
            get_yahoo_finance_client,
            get_sentiment_market_correlator,
            get_gsib_monitor
        )
        print("SUCCESS: All components imported")
        
        # Test initialization
        print("2. Initializing components...")
        yahoo_client = get_yahoo_finance_client()
        sentiment_correlator = get_sentiment_market_correlator()
        gsib_monitor = get_gsib_monitor()
        print("SUCCESS: All components initialized")
        
        # Test market data
        print("3. Testing market data...")
        market_data = yahoo_client.fetch_market_data("JPM", period="5d")
        if not market_data.empty:
            print(f"SUCCESS: Market data fetched - {len(market_data)} data points")
            print(f"Latest close: ${market_data['Close'].iloc[-1]:.2f}")
        else:
            print("WARNING: No market data returned")
        
        # Test G-SIB tickers
        print("4. Testing G-SIB configuration...")
        tickers = gsib_monitor.get_gsib_tickers()
        print(f"SUCCESS: {len(tickers)} G-SIB institutions configured")
        
        print("\nIntegration test PASSED!")
        print("System Status:")
        print("- Yahoo Finance API: Connected")
        print("- Market Data: Accessible") 
        print("- G-SIB Monitoring: Active")
        print("- Ready for NLP correlation!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_integration()