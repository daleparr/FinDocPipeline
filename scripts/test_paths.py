"""Test Python paths and imports."""
import sys
import os

print("\n=== Python Path ===")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n=== Current Working Directory ===")
print(os.getcwd())

print("\n=== Directory Contents ===")
for item in os.listdir('.'):
    print(f"- {item}")

print("\n=== Testing Imports ===")
try:
    import pandas
    print("✓ pandas:", pandas.__version__)
except ImportError as e:
    print("✗ pandas:", e)

try:
    import openpyxl
    print("✓ openpyxl:", openpyxl.__version__)
except ImportError as e:
    print("✗ openpyxl:", e)

try:
    from etl.parsers.excel_parser import ExcelParser
    print("✓ ExcelParser imported successfully")
except ImportError as e:
    print("✗ ExcelParser import failed:", e)
    
    # Try to find the module
    print("\n=== Searching for etl package ===")
    for root, dirs, files in os.walk('..'):
        if 'etl' in dirs and 'parsers' in os.listdir(os.path.join(root, 'etl')):
            print(f"Found etl package at: {os.path.abspath(os.path.join(root, 'etl'))}")
            print("Contents of parsers directory:")
            try:
                for item in os.listdir(os.path.join(root, 'etl', 'parsers')):
                    print(f"- {item}")
            except Exception as e:
                print(f"Error listing directory: {e}")
