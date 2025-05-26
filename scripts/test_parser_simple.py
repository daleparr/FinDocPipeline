"""Simple test of Excel parser functionality."""
import sys
import os
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create a simple test file
def create_test_excel(file_path):
    """Create a simple Excel file for testing."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Sheet"
    ws.append(["Name", "Value"])
    ws.append(["Test 1", 100])
    ws.append(["Test 2", 200])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(file_path)
    print(f"Created test file: {file_path}")

def main():
    """Run the test."""
    test_file = Path("test_excel.xlsx")
    
    try:
        # Create test file
        create_test_excel(test_file)
        
        # Import and test the parser
        from src.etl.parsers.excel_parser import ExcelParser
        
        print("Testing ExcelParser...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        result = parser.parse(test_file)
        
        print("\nParser result:")
        print(json.dumps(result, indent=2, default=str))
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
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
