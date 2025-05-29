#!/usr/bin/env python3
"""
Bank of England Mosaic Lens Setup
A comprehensive financial intelligence platform with Market Intelligence & G-SIB Monitoring.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Bank of England Mosaic Lens - Financial Intelligence Platform"

# Read version from VERSION file
def read_version():
    version_path = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_path):
        with open(version_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "2.2.0"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="bank-of-england-mosaic-lens",
    version=read_version(),
    author="Bank of England Intelligence Team",
    author_email="intelligence-team@bankofengland.co.uk",
    description="A comprehensive financial intelligence platform with Market Intelligence & G-SIB Monitoring",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/daleparr/Bank-of-England-Mosaic-Lens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "yfinance>=0.2.18",
        "scikit-learn>=1.3.0",
        "pandas-market-calendars>=4.3.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.1",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "market-intelligence": [
            "yfinance>=0.2.18",
            "pandas-market-calendars>=4.3.0",
            "scikit-learn>=1.3.0",
        ],
        "nlp": [
            "transformers>=4.21.0",
            "torch>=1.12.0",
            "nltk>=3.8",
            "spacy>=3.4.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "yfinance>=0.2.18",
            "pandas-market-calendars>=4.3.0",
            "scikit-learn>=1.3.0",
            "transformers>=4.21.0",
            "torch>=1.12.0",
            "nltk>=3.8",
            "spacy>=3.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "boe-mosaic-lens=main_dashboard:main",
            "boe-market-intelligence=launch_market_intelligence_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": [
            "etl/data/*.json",
            "etl/data/*.yml",
            "etl/config/*.yaml",
            "etl/config/*.yml",
            "market_intelligence/*.py",
        ],
        "config": [
            "*.yaml",
            "*.yml",
        ],
        "data_science": [
            "config/*.yaml",
            "scripts/**/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "financial-intelligence", "market-intelligence", "g-sib", "banking", 
        "regulatory", "supervision", "risk-assessment", "earnings-analysis",
        "sentiment-analysis", "correlation-analysis", "systemic-risk"
    ],
    project_urls={
        "Bug Reports": "https://github.com/daleparr/Bank-of-England-Mosaic-Lens/issues",
        "Source": "https://github.com/daleparr/Bank-of-England-Mosaic-Lens",
        "Documentation": "https://github.com/daleparr/Bank-of-England-Mosaic-Lens/blob/main/README.md",
        "Release Notes": "https://github.com/daleparr/Bank-of-England-Mosaic-Lens/blob/main/RELEASE_NOTES_v2.2.0.md",
    },
)