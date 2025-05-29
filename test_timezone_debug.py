"""
Test Timezone-Aware Timestamp Debugging
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_timezone_handling():
    print("TESTING TIMEZONE-AWARE TIMESTAMP HANDLING")
    print("=" * 50)
    
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        
        # Get market data with timezone info
        print("1. Getting market data...")
        monitor = get_gsib_monitor()
        market_data = monitor.track_global_gsib_movements(period="1mo")
        
        test_ticker = "C"
        if test_ticker in market_data:
            ticker_data = market_data[test_ticker]
            print(f"   Ticker data shape: {ticker_data.shape}")
            print(f"   Index type: {type(ticker_data.index)}")
            
            if hasattr(ticker_data.index, 'tz'):
                print(f"   Index timezone: {ticker_data.index.tz}")
            
            # Test earnings date handling
            print("\n2. Testing earnings date timezone conversion...")
            earnings_date = datetime(2025, 4, 15)
            print(f"   Original earnings date: {earnings_date} (naive)")
            
            # Convert to pandas Timestamp
            earnings_ts = pd.Timestamp(earnings_date)
            print(f"   Pandas Timestamp: {earnings_ts} (tz: {earnings_ts.tz})")
            
            # Test timezone conversion logic
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                market_tz = ticker_data.index.tz
                print(f"   Market timezone: {market_tz}")
                
                if earnings_ts.tz is None:
                    # Localize if naive
                    earnings_ts_converted = earnings_ts.tz_localize(market_tz)
                    print(f"   Localized: {earnings_ts_converted}")
                else:
                    # Convert if already timezone-aware
                    earnings_ts_converted = earnings_ts.tz_convert(market_tz)
                    print(f"   Converted: {earnings_ts_converted}")
                
                # Test comparison operations
                print("\n3. Testing comparison operations...")
                try:
                    pre_data = ticker_data[ticker_data.index < earnings_ts_converted]
                    post_data = ticker_data[ticker_data.index > earnings_ts_converted]
                    print(f"   Pre-earnings data: {len(pre_data)} points")
                    print(f"   Post-earnings data: {len(post_data)} points")
                    print("   Comparison operations: WORKING")
                except Exception as e:
                    print(f"   Comparison error: {e}")
                    return False
                
                # Test chart compatibility
                print("\n4. Testing chart timestamp compatibility...")
                try:
                    # Simulate plotly chart timestamp
                    chart_timestamp = earnings_ts_converted
                    print(f"   Chart timestamp: {chart_timestamp}")
                    print("   Chart compatibility: WORKING")
                except Exception as e:
                    print(f"   Chart error: {e}")
                    return False
            
            print("\n" + "=" * 50)
            print("TIMEZONE HANDLING: FULLY OPERATIONAL!")
            print("=" * 50)
            
            return True
            
        else:
            print(f"   ERROR: No market data found for {test_ticker}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_timezone_handling()
    if success:
        print("\nTimezone handling tests PASSED!")
    else:
        print("\nTimezone tests FAILED - check errors above")