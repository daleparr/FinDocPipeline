#!/usr/bin/env python3
"""
Bank of England Mosaic Lens v2.1.0 - Main Dashboard Launcher
============================================================

Quick launcher for the Bank of England Supervisor Risk Assessment Dashboard
with integrated technical validation capabilities.

Usage:
    python launch_dashboard.py [--port PORT] [--host HOST]

Examples:
    python launch_dashboard.py                    # Default: localhost:8505
    python launch_dashboard.py --port 8080       # Custom port
    python launch_dashboard.py --host 0.0.0.0    # External access
"""

import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Launch the Bank of England Supervisor Dashboard"""
    
    parser = argparse.ArgumentParser(
        description="Launch Bank of England Mosaic Lens Dashboard v2.1.0"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8505, 
        help="Port to run the dashboard on (default: 8505)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Check if main_dashboard.py exists
    dashboard_path = Path("main_dashboard.py")
    if not dashboard_path.exists():
        print("❌ Error: main_dashboard.py not found in current directory")
        print("Please run this script from the repository root directory")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        "streamlit", "run", "main_dashboard.py",
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--browser.gatherUsageStats", "false"
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    print("Bank of England Mosaic Lens v2.1.0")
    print("=" * 50)
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    print("Features: Risk Analysis + Technical Validation")
    print("Statistical validation with real-time processing")
    print("=" * 50)
    print("Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        # Launch streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()