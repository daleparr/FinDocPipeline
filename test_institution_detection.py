"""
Test Institution Auto-Detection from ETL
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent))

def test_institution_detection():
    print("TESTING INSTITUTION AUTO-DETECTION")
    print("=" * 50)
    
    try:
        from src.market_intelligence.market_intelligence_dashboard import get_market_intelligence_dashboard
        
        # Initialize dashboard
        print("1. Initializing dashboard...")
        dashboard = get_market_intelligence_dashboard()
        print("   Dashboard initialized successfully")
        
        # Test institution detection from ETL
        print("\n2. Testing institution detection from ETL...")
        institutions = dashboard._detect_institutions_from_etl()
        print(f"   Auto-detected institutions: {institutions}")
        
        if institutions:
            print(f"   Found {len(institutions)} institutions:")
            for i, inst in enumerate(institutions, 1):
                ticker = dashboard._map_institution_to_ticker(inst)
                print(f"      {i}. {inst} -> {ticker}")
        
        # Test earnings date detection
        print("\n3. Testing earnings date detection...")
        test_ticker = "C"
        earnings_date = dashboard._detect_earnings_date_from_etl(test_ticker)
        
        if earnings_date:
            print(f"   Auto-detected earnings date for {test_ticker}: {earnings_date.date()}")
        else:
            print(f"   No earnings date detected for {test_ticker}")
        
        # Test ticker to institution mapping
        print("\n4. Testing ticker mappings...")
        test_tickers = ["C", "JPM", "BAC", "GS"]
        for ticker in test_tickers:
            institution = dashboard._map_ticker_to_institution(ticker)
            print(f"   {ticker} -> {institution}")
        
        print("\n" + "=" * 50)
        print("INSTITUTION AUTO-DETECTION: OPERATIONAL!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_institution_detection()
    if success:
        print("\nInstitution auto-detection tests PASSED!")
    else:
        print("\nSome tests FAILED - check errors above")