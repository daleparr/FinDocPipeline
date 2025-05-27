"""
Simple Dashboard Launcher

Launch the enhanced stakeholder risk assessment dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the stakeholder dashboard"""
    dashboard_path = Path(__file__).parent / "stakeholder_risk_dashboard.py"
    
    print("=" * 60)
    print("LAUNCHING STAKEHOLDER RISK ASSESSMENT DASHBOARD")
    print("=" * 60)
    print(f"Dashboard: {dashboard_path}")
    print("Features:")
    print("* Phase 4 Business Intelligence Integration")
    print("* Drag & Drop Document Upload")
    print("* Real-time Risk Analysis")
    print("* Business-Friendly Risk Classification")
    print("* Actionable Recommendations")
    print("* Executive Summary Export")
    print("=" * 60)
    print("Opening in your browser...")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Launch Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.headless", "false",
            "--server.port", "8502",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()