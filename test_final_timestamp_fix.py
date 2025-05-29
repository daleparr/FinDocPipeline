"""
Final Test for All Timestamp Arithmetic Fixes
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_final_timestamp_fixes():
    print("TESTING FINAL TIMESTAMP ARITHMETIC FIXES")
    print("=" * 50)
    
    try:
        from src.market_intelligence.sentiment_market_correlator import get_sentiment_market_correlator
        from src.market_intelligence.market_intelligence_dashboard import get_market_intelligence_dashboard
        
        # Initialize components
        print("1. Initializing components...")
        correlator = get_sentiment_market_correlator()
        dashboard = get_market_intelligence_dashboard()
        print("   Components initialized successfully")
        
        # Test sentiment correlator timestamp operations
        print("\n2. Testing sentiment correlator timestamp operations...")
        
        # Test the generate_mock_nlp_signals method (which uses the fixed timestamp arithmetic)
        try:
            mock_signals = correlator.generate_mock_nlp_signals("TestBank", "Q1 2025")
            print(f"   Mock NLP signals generated: {len(mock_signals)} signals")
            if len(mock_signals) > 0:
                print(f"   Sample signal date: {mock_signals.iloc[0]['date']}")
            print("   Sentiment correlator timestamp operations: WORKING")
        except Exception as e:
            print(f"   Sentiment correlator error: {e}")
            return False
        
        # Test dashboard auto-detection methods
        print("\n3. Testing dashboard timestamp operations...")
        try:
            institutions = dashboard._detect_institutions_from_etl()
            print(f"   Auto-detected institutions: {institutions}")
            
            if institutions:
                test_ticker = dashboard._map_institution_to_ticker(institutions[0])
                earnings_date = dashboard._detect_earnings_date_from_etl(test_ticker)
                if earnings_date:
                    print(f"   Auto-detected earnings date: {earnings_date}")
                else:
                    print("   No earnings date detected (expected for test)")
            
            print("   Dashboard timestamp operations: WORKING")
        except Exception as e:
            print(f"   Dashboard error: {e}")
            return False
        
        # Test earnings impact analysis with comprehensive data
        print("\n4. Testing comprehensive earnings impact analysis...")
        earnings_date = datetime(2025, 4, 15)
        
        # Create comprehensive mock data
        mock_data = []
        base_date = pd.Timestamp(earnings_date) - pd.Timedelta(days=10)
        
        for i in range(20):  # 20 days of data
            date = base_date + pd.Timedelta(days=i)
            mock_data.append({
                'date': date,
                'sentiment_score': 0.1 + (i * 0.02) + (0.1 if i > 10 else -0.1),
                'daily_return': 0.001 * (i - 10) + (0.005 if i > 10 else -0.005),
                'risk_alert': 1 if abs(i - 10) <= 3 else 0
            })
        
        merged_df = pd.DataFrame(mock_data)
        print(f"   Created comprehensive mock data with {len(merged_df)} records")
        
        # Test the earnings impact analysis
        impact_analysis = correlator.analyze_earnings_impact(
            merged_df, 
            earnings_date,
            pre_days=7,
            post_days=5
        )
        
        if impact_analysis:
            print("   Comprehensive earnings impact analysis: WORKING")
            print(f"   Analysis components: {list(impact_analysis.keys())}")
            
            if 'pre_earnings' in impact_analysis and 'post_earnings' in impact_analysis:
                pre = impact_analysis['pre_earnings']
                post = impact_analysis['post_earnings']
                print(f"   Pre-earnings avg sentiment: {pre.get('avg_sentiment', 0):.3f}")
                print(f"   Post-earnings avg sentiment: {post.get('avg_sentiment', 0):.3f}")
                
                if 'comparison' in impact_analysis:
                    comp = impact_analysis['comparison']
                    print(f"   Sentiment change: {comp.get('sentiment_change', 0):.3f}")
                    
        else:
            print("   Earnings impact analysis: FAILED")
            return False
        
        print("\n" + "=" * 50)
        print("ALL TIMESTAMP FIXES: FULLY OPERATIONAL!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_timestamp_fixes()
    if success:
        print("\nAll final timestamp fixes PASSED!")
    else:
        print("\nSome fixes FAILED - check errors above")