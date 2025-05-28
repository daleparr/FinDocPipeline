# CELL 1: SETUP AND IMPORTS
# ========================
# Bank of England Mosaic Lens - NLP Validation Setup
# Copy this entire cell into the first cell of your Google Colab notebook

import re
import time
import math
from datetime import datetime
from collections import Counter, defaultdict

# Try to import pandas for real data loading
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    print("✅ pandas available - can load CSV/Excel files")
except ImportError:
    PANDAS_AVAILABLE = False
    print("❌ pandas not available - install with: !pip install pandas")

def load_csv_data(file_path_or_url):
    """
    Load CSV data from file path or URL.
    
    Args:
        file_path_or_url (str): Path to CSV file or URL
        
    Returns:
        list: List of text documents extracted from CSV
    """
    if not PANDAS_AVAILABLE:
        print("pandas required for CSV loading. Install with: !pip install pandas")
        return []
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path_or_url)
        
        # Extract text from all string columns
        documents = []
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                for value in df[col].dropna():
                    if isinstance(value, str) and len(value) > 50:  # Meaningful text
                        documents.append(value)
        
        print(f"✅ Loaded {len(documents)} text documents from CSV")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        return []

def load_excel_data(file_path_or_url, sheet_name=0):
    """
    Load Excel data from file path or URL.
    
    Args:
        file_path_or_url (str): Path to Excel file or URL
        sheet_name (str/int): Sheet name or index to read
        
    Returns:
        list: List of text documents extracted from Excel
    """
    if not PANDAS_AVAILABLE:
        print("pandas required for Excel loading. Install with: !pip install pandas openpyxl")
        return []
    
    try:
        # Read Excel file
        df = pd.read_excel(file_path_or_url, sheet_name=sheet_name)
        
        # Extract text from all string columns
        documents = []
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                for value in df[col].dropna():
                    if isinstance(value, str) and len(value) > 50:  # Meaningful text
                        documents.append(value)
        
        print(f"✅ Loaded {len(documents)} text documents from Excel")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading Excel: {str(e)}")
        return []

print("=" * 80)
print("BANK OF ENGLAND MOSAIC LENS - NLP VALIDATION")
print("=" * 80)
print(f"Validation started at: {datetime.now()}")
print("Purpose: Code scrutiny and NLP workflow demonstration")
print("Libraries: Core Python + optional pandas for real data")
print("=" * 80)

# Example usage (uncomment to load real data):
# real_documents = load_csv_data('your_file.csv')
# real_documents = load_excel_data('your_file.xlsx', sheet_name='Sheet1')