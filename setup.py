from setuptools import setup, find_packages

setup(
    name="bank-etl",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "transformers>=4.15.0",
        "torch>=1.9.0",
        "bertopic>=0.12.0",
        "sentence-transformers>=2.2.0",
        "spacy>=3.1.0",
        "nltk>=3.6.3",
        "shap>=0.41.0",
        "lime>=0.2.0.1",
        "streamlit>=0.95.0",
        "plotly>=5.3.0",
        "dash>=2.0.0",
        "slack-sdk>=3.10.0",
        "python-dotenv>=0.19.0",
        "pyarrow>=4.0.0",
        "pdfplumber>=0.9.0",
        "python-pptx>=0.6.21"
    ],
    python_requires=">=3.8",
    author="",
    description="ETL pipeline for bank earnings call data processing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'run-etl = run_etl:main',
        ],
    },
)
