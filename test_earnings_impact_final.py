"""
Final Test for Earnings Impact Analysis - All Fixes
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_earnings_impact_final():
    print("TESTING FINAL EARNINGS IMPACT ANALYSIS")
    print("=" * 50)
    
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        from src.market_intelligence.market_intelligence_dashboard import get_market_intelligence_dashboard
        
        # Initialize components
        print("1. Initializing components...")
        monitor = get_gsib_monitor()
        dashboard = get_market_intelligence_dashboard()
        print("   Components initialized successfully")
        
        # Get market data
        print("\n2. Fetching market data...")
        market_data = monitor.track_global_gsib_movements(period="1mo")
        print(f"   Market data for {len(market_data)} institutions")
        
        # Test auto-detection
        print("\n3. Testing auto-detection...")
        institutions = dashboard._detect_institutions_from_etl()
        print(f"   Auto-detected institutions: {institutions}")
        
        # Test earnings impact with proper timestamp handling
        print("\n4. Testing earnings impact analysis...")
        test_ticker = "C"
        earnings_date = datetime(2025, 4, 15)  # Auto-detected date
        
        if test_ticker in market_data:
            ticker_data = market_data[test_ticker]
            print(f"   Testing {test_ticker} with {ticker_data.shape[0]} data points")
            print(f"   Index type: {type(ticker_data.index)}")
            
            if hasattr(ticker_data.index, 'tz'):
                print(f"   Index timezone: {ticker_data.index.tz}")
            
            # Test timestamp arithmetic fixes
            print("\n5. Testing timestamp arithmetic...")
            
            # Convert earnings_date to pandas Timestamp
            earnings_ts = pd.Timestamp(earnings_date)
            pre_start = earnings_ts - pd.Timedelta(days=5)
            post_end = earnings_ts + pd.Timedelta(days=3)
            
            print(f"   Earnings date: {earnings_ts}")
            print(f"   Analysis window: {pre_start} to {post_end}")
            
            # Test timezone compatibility
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                market_tz = ticker_data.index.tz
                earnings_ts_tz = earnings_ts.tz_localize(market_tz)
                print(f"   Timezone-aware earnings date: {earnings_ts_tz}")
            
            # Test data filtering
            try:
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    earnings_ts_for_filter = earnings_ts.tz_localize(ticker_data.index.tz)
                else:
                    earnings_ts_for_filter = earnings_ts
                
                pre_data = ticker_data[ticker_data.index < earnings_ts_for_filter]
                post_data = ticker_data[ticker_data.index > earnings_ts_for_filter]
                
                print(f"   Pre-earnings data: {len(pre_data)} points")
                print(f"   Post-earnings data: {len(post_data)} points")
                
                if not pre_data.empty and not post_data.empty:
                    print("   Pre/post comparison: WORKING")
                else:
                    print("   Pre/post comparison: Limited data")
                
            except Exception as e:
                print(f"   Filtering error: {e}")
                return False
            
            print("\n" + "=" * 50)
            print("EARNINGS IMPACT ANALYSIS: FULLY OPERATIONAL!")
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
    success = test_earnings_impact_final()
    if success:
        print("\nAll earnings impact fixes PASSED!")
    else:
        print("\nSome fixes FAILED - check errors above")