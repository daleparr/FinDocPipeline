"""Direct test of ExcelParser from project root."""
import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    """Run the test."""
    try:
        # Create a test Excel file
        test_file = Path("test_excel.xlsx")
        logger.info("Creating test Excel file: %s", test_file.absolute())
        
        from openpyxl import Workbook
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
            
        # Import and test the parser
        logger.info("Importing ExcelParser...")
        try:
            # Try relative import first
            from src.etl.parsers.excel_parser import ExcelParser
            logger.info("Imported ExcelParser from src.etl.parsers.excel_parser")
        except ImportError as e:
            logger.error("Failed to import ExcelParser: %s", e)
            return 1
        
        logger.info("Creating parser instance...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        
        logger.info("Parsing Excel file...")
        result = parser.parse(test_file)
        
        # Print results
        print("\n=== Parser Result ===")
        print(f"Bank: {result.get('bank')}")
        print(f"Quarter: {result.get('quarter')}")
        print(f"Document Type: {result.get('document_type')}")
        
        if 'content' in result and 'tables' in result['content']:
            print("\nTables found:")
            for sheet_name, sheet_data in result['content']['tables'].items():
                print(f"\nSheet: {sheet_name}")
                print(f"Rows: {sheet_data.get('shape', {}).get('rows')}")
                print(f"Columns: {sheet_data.get('shape', {}).get('columns')}")
                
                if 'columns' in sheet_data:
                    print("Columns:", [col.get('name') for col in sheet_data['columns']])
                
                if 'data' in sheet_data and sheet_data['data']:
                    print("First row of data:", sheet_data['data'][0])
        
        return 0
        
    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        return 1
    finally:
        # Clean up
        try:
            if 'test_file' in locals() and test_file.exists():
                os.remove(test_file)
                logger.info("Cleaned up test file")
        except Exception as e:
            logger.error("Error cleaning up: %s", str(e))

if __name__ == "__main__":
    sys.exit(main())
