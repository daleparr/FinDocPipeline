"""Direct test of Excel parser functionality."""
import sys
import os
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import required modules
try:
    logger.info("Attempting to import openpyxl...")
    from openpyxl import Workbook
    from openpyxl.styles import Font
    logger.info("Successfully imported openpyxl")
    
    logger.info("Attempting to import ExcelParser...")
    from src.etl.parsers.excel_parser import ExcelParser
    logger.info("Successfully imported ExcelParser")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error(f"Python path: {sys.path}")
    sys.exit(1)

def main():
    """Run the direct test."""
    try:
        # Create a test Excel file
        test_file = project_root / "direct_test.xlsx"
        
        logger.info(f"Creating test Excel file: {test_file}")
        wb = Workbook()
        ws = wb.active
        ws.title = "Test Data"
        ws.append(["Name", "Value"])
        ws.append(["Test 1", 100])
        ws.append(["Test 2", 200])
        wb.save(test_file)
        
        if not test_file.exists():
            logger.error("Failed to create test file")
            return 1
            
        logger.info("Test file created successfully")
        
        # Test the ExcelParser
        logger.info("Creating ExcelParser instance...")
        parser = ExcelParser(bank="test_bank", quarter="Q2_2025")
        
        logger.info("Parsing Excel file...")
        result = parser.parse(test_file)
        
        # Print results
        print("\n=== Parser Results ===")
        print(f"Bank: {result.get('bank', 'N/A')}")
        print(f"Quarter: {result.get('quarter', 'N/A')}")
        print(f"Document Type: {result.get('document_type', 'N/A')}")
        
        if 'content' in result and 'tables' in result['content']:
            print("\nTables found:")
            for sheet_name, sheet_data in result['content']['tables'].items():
                print(f"\nSheet: {sheet_name}")
                print(f"Rows: {sheet_data.get('shape', {}).get('rows', 'N/A')}")
                print(f"Columns: {sheet_data.get('shape', {}).get('columns', 'N/A')}")
                
                if 'columns' in sheet_data:
                    print("Columns:", [col.get('name', 'N/A') for col in sheet_data['columns']])
                
                if 'data' in sheet_data and sheet_data['data']:
                    print("Sample data (first 2 rows):")
                    for i, row in enumerate(sheet_data['data'][:2]):
                        print(f"  Row {i+1}: {row}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        return 1
    finally:
        # Clean up
        try:
            if 'test_file' in locals() and test_file.exists():
                os.remove(test_file)
                logger.info("Cleaned up test file")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

if __name__ == "__main__":
    sys.exit(main())
