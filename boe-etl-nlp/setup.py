#!/usr/bin/env python3
"""
Setup script for boe-etl-nlp package.

This package provides NLP extensions for the boe-etl core package,
including topic modeling, sentiment analysis, and advanced text processing.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')

setup(
    name="boe-etl-nlp",
    version="1.0.0",
    author="Bank of England ETL Team",
    author_email="etl-team@bankofengland.co.uk",
    description="NLP extensions for the BoE ETL pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daleparr/boe-etl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "boe-etl>=1.0.0",  # Core dependency
        "bertopic>=0.15.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "umap-learn>=0.5.3",
        "hdbscan>=0.8.29",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "wordcloud>=1.9.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "wordcloud>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "boe-etl-nlp=boe_etl_nlp.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "boe_etl_nlp": [
            "data/*.json",
            "data/*.yml",
            "config/*.yaml",
        ],
    },
)