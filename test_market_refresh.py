"""
Test Market Data Refresh Process
Debug the correlation analysis issue
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_market_refresh():
    print("Testing Market Data Refresh Process...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        from src.market_intelligence.sentiment_market_correlator import get_sentiment_market_correlator
        print("SUCCESS: Components imported")
        
        # Initialize components
        print("2. Initializing components...")
        monitor = get_gsib_monitor()
        correlator = get_sentiment_market_correlator()
        print("SUCCESS: Components initialized")
        
        # Test G-SIB tracking
        print("3. Testing G-SIB tracking...")
        market_data = monitor.track_global_gsib_movements(period="5d")
        print(f"Market data returned: {len(market_data)} institutions")
        
        if market_data:
            print("Sample institutions with data:")
            for ticker, data in list(market_data.items())[:3]:
                print(f"  - {ticker}: {len(data)} data points")
        
        # Test correlation analysis
        print("4. Testing correlation analysis...")
        if market_data:
            correlation_analysis = monitor.detect_cross_market_correlations(market_data)
            print(f"Correlation analysis keys: {list(correlation_analysis.keys())}")
            
            if 'correlation_matrix' in correlation_analysis:
                corr_matrix = correlation_analysis['correlation_matrix']
                print(f"Correlation matrix shape: {corr_matrix.shape}")
            else:
                print("WARNING: No correlation matrix found")
        else:
            print("WARNING: No market data available for correlation analysis")
        
        # Test systemic risk calculation
        print("5. Testing systemic risk calculation...")
        if market_data:
            systemic_risk = monitor.calculate_systemic_risk_score(market_data)
            print(f"Systemic risk keys: {list(systemic_risk.keys())}")
        
        print("\nMarket refresh test completed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Market refresh test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_market_refresh()