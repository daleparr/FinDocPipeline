"""Direct test of ExcelParser with detailed logging."""
import sys
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_excel_parser():
    """Test the ExcelParser directly."""
    try:
        # Create a test Excel file
        test_file = Path("direct_test.xlsx")
        logger.info(f"Creating test file: {test_file}")
        
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
            return False
            
        logger.info("Test file created successfully")
        
        # Import and test the parser
        logger.info("Importing ExcelParser...")
        from src.etl.parsers.excel_parser import ExcelParser
        
        logger.info("Creating parser instance...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        
        logger.info("Parsing Excel file...")
        result = parser.parse(test_file)
        
        logger.info("Parse completed. Result keys: %s", list(result.keys()))
        
        # Print result structure
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
                    
                    if 'columns' in sheet_data:
                        print("Columns:", [col.get('name') for col in sheet_data['columns']])
                    
                    if 'data' in sheet_data and sheet_data['data']:
                        print("First row of data:", sheet_data['data'][0])
        
        return True
        
    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        return False
    finally:
        # Clean up
        try:
            if 'test_file' in locals() and test_file.exists():
                os.remove(test_file)
                logger.info("Cleaned up test file")
        except Exception as e:
            logger.error("Error cleaning up: %s", str(e))

def main():
    """Run the test."""
    logger.info("Starting Excel parser test...")
    success = test_excel_parser()
    
    if success:
        logger.info("✅ Test completed successfully")
        return 0
    else:
        logger.error("❌ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
