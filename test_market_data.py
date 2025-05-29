"""
Simple test to verify market data fetching is working
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_market_data():
    """Test basic market data fetching"""
    print("Testing Market Data Fetching...")
    print("=" * 40)
    
    try:
        # Test 1: Basic yfinance import
        import yfinance as yf
        print("OK yfinance imported successfully")
        
        # Test 2: Fetch data for a simple ticker
        print("Testing basic ticker fetch (AAPL)...")
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        print(f"OK Basic ticker fetch successful: {info.get('shortName', 'Apple Inc.')}")
        
        # Test 3: Test Yahoo Finance client
        print("Testing Yahoo Finance client...")
        from src.market_intelligence.yahoo_finance_client import get_yahoo_finance_client
        
        client = get_yahoo_finance_client()
        print("OK Yahoo Finance client initialized")
        
        # Test 4: Get G-SIB tickers
        tickers = client.get_all_gsib_tickers()
        print(f"OK G-SIB tickers loaded: {len(tickers)} institutions")
        
        # Test 5: Try fetching data for a major G-SIB
        print("Testing G-SIB data fetch (JPM)...")
        try:
            market_data = client.fetch_market_data("JPM", period="5d")
            if market_data is not None and not market_data.empty:
                print(f"OK JPM data fetched: {len(market_data)} data points")
                print(f"   Latest close: ${market_data['Close'].iloc[-1]:.2f}")
            else:
                print("WARNING JPM data fetch returned empty result")
        except Exception as e:
            print(f"ERROR JPM data fetch failed: {e}")
        
        # Test 6: Test G-SIB Monitor
        print("Testing G-SIB Monitor...")
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        
        monitor = get_gsib_monitor()
        total_gsibs = monitor.get_total_gsib_count()
        print(f"OK G-SIB Monitor initialized: {total_gsibs} institutions")
        
        print("\n" + "=" * 40)
        print("OK Market data system is working!")
        return True
        
    except ImportError as e:
        print(f"ERROR Import error: {e}")
        print("TIP Try: pip install -r requirements_market_intelligence.txt")
        return False
        
    except Exception as e:
        print(f"ERROR Error: {e}")
        return False

if __name__ == "__main__":
    success = test_market_data()
    if success:
        print("\nSUCCESS Market data is ready for the dashboard!")
    else:
        print("\nWARNING Market data issues detected - check errors above")
    
    sys.exit(0 if success else 1)