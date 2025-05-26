"""Debug script for Excel parser."""
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

def test_imports():
    """Test importing required modules."""
    try:
        logger.info("Testing imports...")
        import pandas as pd
        import openpyxl
        from openpyxl import Workbook
        logger.info("✓ Successfully imported pandas and openpyxl")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_excel_creation():
    """Test creating an Excel file."""
    try:
        logger.info("Testing Excel file creation...")
        from openpyxl import Workbook
        test_file = Path("test_excel.xlsx")
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Test Sheet"
        ws.append(["Name", "Value"])
        ws.append(["Test 1", 100])
        ws.append(["Test 2", 200])
        wb.save(test_file)
        
        if test_file.exists():
            logger.info(f"✓ Successfully created {test_file}")
            return test_file
        else:
            logger.error("✗ Failed to create test file")
            return None
            
    except Exception as e:
        logger.error(f"✗ Excel creation failed: {e}")
        return None

def test_parser():
    """Test the Excel parser."""
    test_file = test_excel_creation()
    if not test_file:
        return False
    
    try:
        logger.info("Testing ExcelParser...")
        from src.etl.parsers.excel_parser import ExcelParser
        
        logger.info("Creating parser instance...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        
        logger.info("Parsing Excel file...")
        result = parser.parse(test_file)
        
        logger.info("Parser completed successfully")
        print("\n=== Parser Result ===")
        print(f"Bank: {result.get('bank')}")
        print(f"Quarter: {result.get('quarter')}")
        print(f"Document Type: {result.get('document_type')}")
        
        if 'content' in result and 'tables' in result['content']:
            print("\nTables:")
            for sheet_name, sheet_data in result['content']['tables'].items():
                print(f"\nSheet: {sheet_name}")
                print(f"Rows: {sheet_data.get('shape', {}).get('rows')}")
                print(f"Columns: {sheet_data.get('shape', {}).get('columns')}")
                
                if 'columns' in sheet_data:
                    print("Column names:", [col.get('name') for col in sheet_data['columns']])
                
                if 'data' in sheet_data and sheet_data['data']:
                    print("First row of data:", sheet_data['data'][0])
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Parser test failed: {e}", exc_info=True)
        return False
    finally:
        # Clean up
        try:
            if test_file and test_file.exists():
                os.remove(test_file)
                logger.info(f"Cleaned up {test_file}")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

def main():
    """Run all tests."""
    logger.info("Starting debug session...")
    
    # Test 1: Basic imports
    if not test_imports():
        logger.error("❌ Failed import tests")
        return 1
    
    # Test 2: Excel creation
    if not test_excel_creation():
        logger.error("❌ Failed Excel creation test")
        return 1
    
    # Test 3: Parser functionality
    if not test_parser():
        logger.error("❌ Failed parser test")
        return 1
    
    logger.info("✅ All tests completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
