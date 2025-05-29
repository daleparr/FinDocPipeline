"""
Debug Correlation Analysis Issue
Check if daily_return column exists in market data
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_correlation_debug():
    print("Debugging Correlation Analysis...")
    
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        
        monitor = get_gsib_monitor()
        
        # Get market data for a few institutions
        print("1. Getting market data...")
        market_data = monitor.track_global_gsib_movements(period="5d")
        
        if market_data:
            # Check first few institutions
            for ticker, data in list(market_data.items())[:3]:
                print(f"\n{ticker} data:")
                print(f"  Shape: {data.shape}")
                print(f"  Columns: {list(data.columns)}")
                
                # Check if daily_return exists
                if 'daily_return' in data.columns:
                    returns = data['daily_return'].dropna()
                    print(f"  Daily returns: {len(returns)} values")
                    print(f"  Sample returns: {returns.head(3).tolist()}")
                else:
                    print("  WARNING: No 'daily_return' column found!")
        
        # Test correlation analysis step by step
        print("\n2. Testing correlation analysis...")
        
        # Extract returns data manually
        returns_data = {}
        for ticker, df in list(market_data.items())[:5]:  # Test with first 5
            if not df.empty and 'daily_return' in df.columns:
                returns_data[ticker] = df['daily_return'].dropna()
                print(f"  {ticker}: {len(returns_data[ticker])} return values")
        
        print(f"\nReturns data available for {len(returns_data)} institutions")
        
        if len(returns_data) >= 2:
            import pandas as pd
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            print(f"Combined returns DataFrame shape: {returns_df.shape}")
            
            if not returns_df.empty:
                correlation_matrix = returns_df.corr()
                print(f"Correlation matrix shape: {correlation_matrix.shape}")
                print("SUCCESS: Correlation analysis should work!")
            else:
                print("ERROR: Returns DataFrame is empty after dropna()")
        else:
            print("ERROR: Insufficient returns data for correlation analysis")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correlation_debug()