"""
Simple verification script for G-SIB coverage
No Unicode characters for Windows compatibility
"""

import sys
from pathlib import Path
import yaml

def verify_gsib_coverage():
    """Verify all G-SIBs are properly configured"""
    print("Verifying G-SIB Coverage Across All Components")
    print("=" * 50)
    
    # 1. Check configuration file
    config_path = Path("config/gsib_institutions.yaml")
    if not config_path.exists():
        print("ERROR: Configuration file not found")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Count G-SIBs in configuration
    total_gsibs = 0
    gsib_data = config.get('global_gsibs', {})
    
    for bucket_name, banks in gsib_data.items():
        if isinstance(banks, list):
            total_gsibs += len(banks)
            print(f"{bucket_name}: {len(banks)} institutions")
    
    print(f"Total G-SIBs in configuration: {total_gsibs}")
    
    # 2. Test Yahoo Finance client
    try:
        sys.path.append(str(Path(__file__).parent))
        from src.market_intelligence.yahoo_finance_client import get_yahoo_finance_client
        
        client = get_yahoo_finance_client()
        all_tickers = client.get_all_gsib_tickers()
        print(f"Yahoo Finance client can access: {len(all_tickers)} G-SIBs")
        
        # Show breakdown by buckets
        buckets = client.get_gsib_by_systemic_importance()
        for bucket, tickers in buckets.items():
            print(f"  {bucket}: {len(tickers)} institutions")
            
    except Exception as e:
        print(f"ERROR testing Yahoo Finance client: {e}")
        return False
    
    # 3. Test G-SIB Monitor
    try:
        from src.market_intelligence.gsib_monitor import get_gsib_monitor
        
        monitor = get_gsib_monitor()
        monitor_count = monitor.get_total_gsib_count()
        print(f"G-SIB Monitor tracking: {monitor_count} institutions")
        
        # Show breakdown by buckets
        buckets = monitor.get_gsib_by_systemic_importance()
        for bucket, tickers in buckets.items():
            print(f"  {bucket}: {len(tickers)} institutions")
            
    except Exception as e:
        print(f"ERROR testing G-SIB Monitor: {e}")
        return False
    
    # 4. Check main dashboard institutions
    try:
        with open('main_dashboard.py', 'r') as f:
            content = f.read()
            
        # Count institutions in main dashboard
        lines = content.split('\n')
        in_institutions_list = False
        institution_count = 0
        
        for line in lines:
            if 'institutions = [' in line:
                in_institutions_list = True
                continue
            if in_institutions_list and ']' in line and 'institutions' not in line:
                break
            if in_institutions_list and '"' in line and ',' in line:
                institution_count += 1
        
        print(f"Main dashboard institutions: {institution_count}")
        
    except Exception as e:
        print(f"ERROR checking main dashboard: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY:")
    print(f"Configuration file: {total_gsibs} G-SIBs")
    print(f"Yahoo Finance client: {len(all_tickers)} G-SIBs")
    print(f"G-SIB Monitor: {monitor_count} G-SIBs")
    print(f"Main dashboard: {institution_count} institutions (includes 'Other')")
    
    # Expected: 33 G-SIBs + 1 "Other" = 34 total in dashboard
    expected_dashboard = 34
    
    if (total_gsibs == 33 and 
        len(all_tickers) == 33 and 
        monitor_count == 33 and 
        institution_count == expected_dashboard):
        print("\nSUCCESS: All components have complete G-SIB coverage!")
        return True
    else:
        print("\nWARNING: Inconsistent G-SIB counts detected")
        return False

if __name__ == "__main__":
    success = verify_gsib_coverage()
    sys.exit(0 if success else 1)