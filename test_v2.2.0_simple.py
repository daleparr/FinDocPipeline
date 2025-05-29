#!/usr/bin/env python3
"""
Bank of England Mosaic Lens v2.2.0 Release Verification Test (Simple)
Comprehensive test to verify all components are ready for release
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_version():
    """Test that VERSION file contains 2.2.0"""
    print("Testing version...")
    try:
        with open('VERSION', 'r') as f:
            version = f.read().strip()
        
        if version == "2.2.0":
            print("PASS: VERSION file: 2.2.0")
            return True
        else:
            print(f"FAIL: VERSION file contains: {version}, expected: 2.2.0")
            return False
    except FileNotFoundError:
        print("FAIL: VERSION file not found")
        return False

def test_release_files():
    """Test that all release files exist"""
    print("\nTesting release files...")
    
    required_files = [
        "RELEASE_NOTES_v2.2.0.md",
        "DEPLOYMENT_GUIDE_v2.2.0.md",
        "CHANGELOG_v2.2.0.md",
        "RELEASE_SUMMARY_v2.2.0.md",
        "requirements_v2.2.0.txt"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"PASS: {file}")
        else:
            print(f"FAIL: Missing: {file}")
            all_exist = False
    
    return all_exist

def test_market_intelligence_components():
    """Test that all market intelligence components can be imported"""
    print("\nTesting market intelligence components...")
    
    components = [
        ("src.market_intelligence.gsib_monitor", "G-SIB Monitor"),
        ("src.market_intelligence.yahoo_finance_client", "Yahoo Finance Client"),
        ("src.market_intelligence.sentiment_market_correlator", "Sentiment Correlator"),
        ("src.market_intelligence.market_indicators", "Market Indicators"),
        ("src.market_intelligence.market_intelligence_dashboard", "Market Intelligence Dashboard")
    ]
    
    all_imported = True
    for module_name, display_name in components:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"PASS: {display_name}")
            else:
                print(f"FAIL: {display_name} - Module not found")
                all_imported = False
        except Exception as e:
            print(f"FAIL: {display_name} - Import error: {e}")
            all_imported = False
    
    return all_imported

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\nTesting dependencies...")
    
    required_deps = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("plotly", "Plotly"),
        ("yfinance", "Yahoo Finance"),
        ("sklearn", "Scikit-learn")
    ]
    
    all_available = True
    for dep_name, display_name in required_deps:
        try:
            __import__(dep_name)
            print(f"PASS: {display_name}")
        except ImportError:
            print(f"FAIL: {display_name} - Not installed")
            all_available = False
    
    return all_available

def test_timestamp_arithmetic():
    """Test that timestamp arithmetic is working correctly"""
    print("\nTesting timestamp arithmetic...")
    
    try:
        import pandas as pd
        from datetime import datetime
        
        # Test pandas Timedelta usage
        base_date = pd.Timestamp.now()
        future_date = base_date + pd.Timedelta(days=5)
        past_date = base_date - pd.Timedelta(days=3)
        
        print("PASS: Pandas Timedelta arithmetic working")
        
        # Test timezone handling
        ny_time = pd.Timestamp.now(tz='America/New_York')
        utc_time = ny_time.tz_convert('UTC')
        
        print("PASS: Timezone conversion working")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Timestamp arithmetic test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("Bank of England Mosaic Lens v2.2.0 Release Verification")
    print("=" * 60)
    
    tests = [
        ("Version Check", test_version),
        ("Release Files", test_release_files),
        ("Market Intelligence Components", test_market_intelligence_components),
        ("Dependencies", test_dependencies),
        ("Timestamp Arithmetic", test_timestamp_arithmetic)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"PASS: {test_name}")
            passed += 1
        else:
            print(f"FAIL: {test_name}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\nALL TESTS PASSED - RELEASE v2.2.0 IS READY!")
        print("Proceed with GitHub release creation")
        return True
    else:
        print(f"\n{failed} TESTS FAILED - REVIEW ISSUES BEFORE RELEASE")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)