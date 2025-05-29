"""
Setup script for Market Intelligence dependencies
Installs required packages for Yahoo Finance integration
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Setup Market Intelligence dependencies"""
    
    print("🔧 Setting up Market Intelligence Dependencies")
    print("=" * 50)
    
    # Required packages
    packages = [
        "yfinance>=0.2.18",
        "pandas>=1.5.0", 
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "urllib3>=2.0.0"
    ]
    
    print(f"📦 Installing {len(packages)} packages...")
    print()
    
    failed_packages = []
    
    for i, package in enumerate(packages, 1):
        print(f"[{i}/{len(packages)}] Installing {package}...")
        
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
        print()
    
    print("=" * 50)
    
    if not failed_packages:
        print("🎉 All dependencies installed successfully!")
        print()
        print("🚀 Ready to launch Market Intelligence Dashboard!")
        print("   Run: python launch_market_intelligence_dashboard.py")
        print("   URL: http://localhost:8514")
    else:
        print(f"⚠️ {len(failed_packages)} packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print()
        print("💡 Try installing manually:")
        print("   pip install -r requirements_market_intelligence.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()