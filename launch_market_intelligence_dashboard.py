"""
Launch script for the upgraded BoE Supervisor Dashboard with Market Intelligence
Runs on port 8514 with the new Yahoo Finance integration
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the upgraded BoE Supervisor Dashboard on port 8514"""
    
    # Set the port
    port = 8514
    
    # Get the path to the main dashboard
    dashboard_path = Path(__file__).parent / "main_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    print("Launching BoE Supervisor Dashboard with Market Intelligence")
    print("=" * 60)
    print(f"Dashboard: {dashboard_path}")
    print(f"Port: {port}")
    print(f"URL: http://localhost:{port}")
    print("=" * 60)
    print("New Features:")
    print("   - Market Intelligence Tab")
    print("   - G-SIB Monitoring")
    print("   - Yahoo Finance Integration")
    print("   - Sentiment-Market Correlation Alerts")
    print("   - Systemic Risk Analysis")
    print("=" * 60)
    
    try:
        # Launch streamlit with the specified port
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print("\nPress Ctrl+C to stop the dashboard")
        print("Opening browser at http://localhost:8514")
        print("\n" + "=" * 60)
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
        print("Thank you for using BoE Supervisor Dashboard!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("   1. Check if streamlit is installed: pip install streamlit")
        print("   2. Install market intelligence dependencies:")
        print("      pip install -r requirements_market_intelligence.txt")
        print("   3. Ensure port 8514 is available")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()