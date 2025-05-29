"""
Simple Test for Timestamp Arithmetic Fixes
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_timestamp_fixes():
    print("TESTING TIMESTAMP ARITHMETIC FIXES")
    print("=" * 50)
    
    try:
        from src.market_intelligence.sentiment_market_correlator import get_sentiment_market_correlator
        
        # Initialize correlator
        print("1. Initializing sentiment correlator...")
        correlator = get_sentiment_market_correlator()
        print("   Correlator initialized successfully")
        
        # Test earnings impact analysis with timestamp arithmetic
        print("\n2. Testing earnings impact analysis...")
        earnings_date = datetime(2025, 4, 15)
        
        # Create mock merged data
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
        print(f"   Created mock data with {len(merged_df)} records")
        
        # Test the fixed earnings impact analysis
        print("   Running analyze_earnings_impact method...")
        impact_analysis = correlator.analyze_earnings_impact(
            merged_df, 
            earnings_date,
            pre_days=5,
            post_days=3
        )
        
        if impact_analysis:
            print("   Earnings impact analysis: WORKING")
            print(f"   Analysis keys: {list(impact_analysis.keys())}")
            
            if 'pre_earnings' in impact_analysis:
                pre = impact_analysis['pre_earnings']
                print(f"   Pre-earnings sentiment: {pre.get('avg_sentiment', 0):.3f}")
            
            if 'post_earnings' in impact_analysis:
                post = impact_analysis['post_earnings']
                print(f"   Post-earnings sentiment: {post.get('avg_sentiment', 0):.3f}")
                
        else:
            print("   Earnings impact analysis: FAILED")
            return False
        
        print("\n" + "=" * 50)
        print("TIMESTAMP FIXES: OPERATIONAL!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_timestamp_fixes()
    if success:
        print("\nTimestamp arithmetic fixes PASSED!")
    else:
        print("\nTimestamp fixes FAILED - check errors above")