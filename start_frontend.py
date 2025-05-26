#!/usr/bin/env python3
"""
ETL Pipeline Frontend Launcher

This script launches the Streamlit web interface for the ETL pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements_frontend.txt")
        return False

def launch_frontend():
    """Launch the Streamlit frontend."""
    if not check_requirements():
        return
    
    print("🚀 Launching ETL Pipeline Frontend...")
    print("📊 The web interface will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "etl_frontend.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped")
    except Exception as e:
        print(f"❌ Error launching frontend: {e}")

if __name__ == "__main__":
    launch_frontend()