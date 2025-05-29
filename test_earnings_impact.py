"""
Test Earnings Impact Analysis
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_earnings_impact():
    print("Testing Earnings Impact Analysis...")
    
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        from src.market_intelligence.sentiment_market_correlator import get_sentiment_market_correlator
        
        # Initialize components
        monitor = get_gsib_monitor()
        correlator = get_sentiment_market_correlator()
        
        # Get market data
        print("1. Getting market data...")
        market_data = monitor.track_global_gsib_movements(period="1mo")
        print(f"Market data for {len(market_data)} institutions")
        
        # Test earnings impact for a specific ticker
        test_ticker = "C"  # Citigroup
        earnings_date = datetime(2025, 4, 28)  # Example earnings date
        
        if test_ticker in market_data:
            ticker_data = market_data[test_ticker]
            print(f"\n2. Testing earnings impact for {test_ticker}...")
            print(f"   Ticker data shape: {ticker_data.shape}")
            print(f"   Earnings date: {earnings_date.date()}")
            
            # Check data around earnings date
            pre_start = earnings_date - timedelta(days=5)
            post_end = earnings_date + timedelta(days=3)
            
            print(f"   Analysis window: {pre_start.date()} to {post_end.date()}")
            
            # Check if we have data in this period
            if hasattr(ticker_data.index, 'date'):
                data_dates = [d.date() for d in ticker_data.index]
            else:
                data_dates = ticker_data.index.tolist()
            
            print(f"   Available data dates: {len(data_dates)} points")
            print(f"   Date range: {min(data_dates)} to {max(data_dates)}")
            
            # Test NLP data loading
            print("\n3. Testing NLP data loading...")
            nlp_signals = correlator.fetch_nlp_signals("Citigroup", "Q1 2025")
            print(f"   NLP signals: {len(nlp_signals)} records")
            
            if not nlp_signals.empty:
                print(f"   Sample topics: {nlp_signals['topic_label'].unique()[:3]}")
            
            print("\nEarnings impact test completed successfully!")
            return True
            
        else:
            print(f"ERROR: No market data found for {test_ticker}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_earnings_impact()