"""
Test All Timestamp Arithmetic Fixes
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_all_timestamp_fixes():
    print("TESTING ALL TIMESTAMP ARITHMETIC FIXES")
    print("=" * 50)
    
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        from src.market_intelligence.sentiment_market_correlator import get_sentiment_market_correlator
        from src.market_intelligence.market_intelligence_dashboard import get_market_intelligence_dashboard
        
        # Initialize components
        print("1. Initializing components...")
        monitor = get_gsib_monitor()
        correlator = get_sentiment_market_correlator()
        dashboard = get_market_intelligence_dashboard()
        print("   All components initialized successfully")
        
        # Get market data
        print("\n2. Getting market data...")
        market_data = monitor.track_global_gsib_movements(period="1mo")
        print(f"   Market data for {len(market_data)} institutions")
        
        # Test earnings impact analysis (this should trigger the fixed timestamp arithmetic)
        print("\n3. Testing earnings impact analysis...")
        test_ticker = "C"
        earnings_date = datetime(2025, 4, 15)
        
        if test_ticker in market_data:
            ticker_data = market_data[test_ticker]
            print(f"   Testing {test_ticker} with {ticker_data.shape[0]} data points")
            
            # Create mock merged data for earnings impact analysis
            mock_data = []
            base_date = earnings_date - pd.Timedelta(days=7)
            
            for i in range(14):  # 2 weeks of data
                date = base_date + pd.Timedelta(days=i)
                mock_data.append({
                    'date': date,
                    'sentiment_score': 0.1 + (i * 0.02),
                    'daily_return': 0.001 * (i - 7),
                    'risk_alert': 1 if abs(i - 7) <= 2 else 0
                })
            
            merged_df = pd.DataFrame(mock_data)
            
            # Test the fixed earnings impact analysis
            print("   Testing analyze_earnings_impact method...")
            impact_analysis = correlator.analyze_earnings_impact(
                merged_df, 
                earnings_date,
                pre_days=5,
                post_days=3
            )
            
            if impact_analysis:
                print("   ✅ Earnings impact analysis: WORKING")
                print(f"      Analysis keys: {list(impact_analysis.keys())}")
            else:
                print("   ❌ Earnings impact analysis: FAILED")
                return False
            
            # Test market indicators
            print("\n4. Testing market indicators...")
            from src.market_intelligence.market_indicators import MarketIndicators
            
            indicators = MarketIndicators()
            try:
                earnings_metrics = indicators.calculate_earnings_impact_metrics(ticker_data, earnings_date)
                print("   ✅ Market indicators earnings metrics: WORKING")
            except Exception as e:
                print(f"   ❌ Market indicators error: {e}")
                return False
            
            # Test yahoo finance client
            print("\n5. Testing yahoo finance client...")
            from src.market_intelligence.yahoo_finance_client import YahooFinanceClient
            
            client = YahooFinanceClient()
            try:
                windows = client.get_earnings_windows([test_ticker], days_before=5, days_after=3)
                print(f"   ✅ Yahoo finance earnings windows: {len(windows)} windows")
            except Exception as e:
                print(f"   ❌ Yahoo finance error: {e}")
                return False
            
            print("\n" + "=" * 50)
            print("ALL TIMESTAMP FIXES: OPERATIONAL!")
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
    success = test_all_timestamp_fixes()
    if success:
        print("\nAll timestamp arithmetic fixes PASSED!")
    else:
        print("\nSome fixes FAILED - check errors above")