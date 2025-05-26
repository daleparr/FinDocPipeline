"""Minimal test for Excel parser."""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run the minimal test."""
    print("Starting minimal Excel parser test...")
    
    # Create test file
    test_file = Path("minimal_test.xlsx")
    try:
        # Create a simple Excel file
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Test Data"
        ws.append(["Name", "Value"])
        ws.append(["Test 1", 100])
        ws.append(["Test 2", 200])
        wb.save(test_file)
        
        if not test_file.exists():
            print("Error: Failed to create test file")
            return 1
            
        print(f"Created test file: {test_file.absolute()}")
        
        # Import and test the parser
        print("\nImporting ExcelParser...")
        from src.etl.parsers.excel_parser import ExcelParser
        
        print("Creating parser instance...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        
        print("Parsing Excel file...")
        result = parser.parse(test_file)
        
        # Print basic result
        print("\n=== Parser Result ===")
        print(f"Bank: {result.get('bank')}")
        print(f"Quarter: {result.get('quarter')}")
        print(f"Document Type: {result.get('document_type')}")
        
        if 'content' in result:
            print("\nContent keys:", list(result['content'].keys()))
            if 'tables' in result['content']:
                print("\nTables found:")
                for sheet_name, sheet_data in result['content']['tables'].items():
                    print(f"\nSheet: {sheet_name}")
                    print(f"Rows: {sheet_data.get('shape', {}).get('rows')}")
                    print(f"Columns: {sheet_data.get('shape', {}).get('columns')}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        try:
            if test_file.exists():
                os.remove(test_file)
                print(f"\nCleaned up {test_file}")
        except Exception as e:
            print(f"Warning: Failed to clean up {test_file}: {e}")

if __name__ == "__main__":
    sys.exit(main())
