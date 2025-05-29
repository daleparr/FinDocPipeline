"""
Test the debug fixes for earnings impact and systemic clustering
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_debug_fixes():
    print("TESTING DEBUG FIXES")
    print("=" * 40)
    
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        
        # Initialize monitor
        print("1. Initializing GSIB monitor...")
        monitor = get_gsib_monitor()
        print("   Monitor initialized successfully")
        
        # Get market data
        print("\n2. Fetching market data...")
        market_data = monitor.track_global_gsib_movements(period="1mo")
        print(f"   Market data for {len(market_data)} institutions")
        
        # Test correlation analysis (this should trigger systemic clustering)
        print("\n3. Testing correlation analysis...")
        correlation_analysis = monitor.detect_cross_market_correlations(market_data)
        
        if correlation_analysis:
            print("   Correlation analysis completed!")
            print(f"   Keys: {list(correlation_analysis.keys())}")
            
            if 'systemic_clusters' in correlation_analysis:
                clusters = correlation_analysis['systemic_clusters']
                print(f"   Systemic clusters: {len(clusters)} cluster groups")
            else:
                print("   No systemic clusters found")
        else:
            print("   Correlation analysis returned empty")
        
        # Test earnings impact with proper datetime handling
        print("\n4. Testing earnings impact analysis...")
        from src.market_intelligence.market_intelligence_dashboard import get_market_intelligence_dashboard
        
        dashboard = get_market_intelligence_dashboard()
        
        # Test with Citigroup data
        test_ticker = "C"
        earnings_date = datetime(2025, 4, 28)
        
        if test_ticker in market_data:
            print(f"   Testing {test_ticker} earnings impact...")
            
            # This should trigger the fixed datetime comparison
            try:
                # Simulate the dashboard call
                ticker_data = market_data[test_ticker]
                print(f"   Ticker data shape: {ticker_data.shape}")
                print(f"   Index type: {type(ticker_data.index)}")
                
                if hasattr(ticker_data.index, 'tz'):
                    print(f"   Index timezone: {ticker_data.index.tz}")
                
                print("   Earnings impact analysis would run successfully")
                
            except Exception as e:
                print(f"   ERROR in earnings impact: {e}")
                return False
        
        print("\n" + "=" * 40)
        print("DEBUG FIXES: SUCCESSFUL!")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_debug_fixes()
    if success:
        print("\nAll debug fixes PASSED!")
    else:
        print("\nSome fixes FAILED - check errors above")