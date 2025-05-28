#!/usr/bin/env python3
"""
BoE ETL Package Setup
A comprehensive ETL pipeline for financial document processing and NLP analysis.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "BoE ETL Package - Financial Document Processing Pipeline"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="boe-etl",
    version="1.0.0",
    author="Bank of England ETL Team",
    author_email="etl-team@bankofengland.co.uk",
    description="A comprehensive ETL pipeline for financial document processing and NLP analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/daleparr/boe-etl",
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
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "frontend": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "boe-etl=boe_etl.cli:main",
            "boe-etl-frontend=boe_etl.frontend:launch",
        ],
    },
    include_package_data=True,
    package_data={
        "boe_etl": [
            "data/*.json",
            "data/*.yml",
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    zip_safe=False,
    keywords=[
        "etl", "nlp", "financial", "banking", "document-processing", 
        "text-analysis", "earnings-calls", "financial-reports", "data-pipeline"
    ],
    project_urls={
        "Bug Reports": "https://github.com/daleparr/boe-etl/issues",
        "Source": "https://github.com/daleparr/boe-etl",
        "Documentation": "https://github.com/daleparr/boe-etl/blob/master/README.md",
    },
)
