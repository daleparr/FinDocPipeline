"""
Minimal test script for Excel parser.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from src.etl.parsers.excel_parser import ExcelParser
    print("Successfully imported all required modules")
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def create_test_file(file_path):
    """Create a simple test Excel file."""
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Test Sheet"
        ws.append(["Column 1", "Column 2", "Column 3"])
        ws.append([1, 2, 3])
        ws.append([4, 5, 6])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(file_path)
        print(f"Created test file: {file_path}")
        return True
    except Exception as e:
        print(f"Error creating test file: {e}")
        return False

def main():
    """Run the test."""
    test_file = project_root / "test_excel.xlsx"
    
    # Create test file
    if not create_test_file(test_file):
        return 1
    
    # Test ExcelParser
    try:
        print("\nTesting ExcelParser...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        result = parser.parse(test_file)
        
        print("\nParser result:")
        print(f"Bank: {result.get('bank')}")
        print(f"Quarter: {result.get('quarter')}")
        print(f"Document Type: {result.get('document_type')}")
        
        if 'content' in result and 'tables' in result['content']:
            print("\nSheets found:")
            for sheet_name, sheet_data in result['content']['tables'].items():
                print(f"- {sheet_name} ({sheet_data.get('shape', {}).get('rows', 0)} rows, {sheet_data.get('shape', {}).get('columns', 0)} columns)")
        
        return 0
    except Exception as e:
        print(f"Error testing ExcelParser: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        try:
            if test_file.exists():
                os.remove(test_file)
                print(f"\nCleaned up test file: {test_file}")
        except Exception as e:
            print(f"Warning: Failed to clean up test file: {e}")

if __name__ == "__main__":
    sys.exit(main())
