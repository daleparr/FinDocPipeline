#!/usr/bin/env python3
"""
Launch Standalone ETL Frontend

Simple launcher that avoids all configuration issues.
"""

import subprocess
import sys

def launch():
    """Launch the standalone frontend."""
    print("🚀 Launching Standalone ETL Frontend...")
    print("📊 Opening at: http://localhost:8503")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "standalone_frontend.py",
            "--server.port", "8503",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    launch()