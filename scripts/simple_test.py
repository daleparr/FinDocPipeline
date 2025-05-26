"""
Simple test script to verify basic Python and package functionality.
"""
import sys
import os
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Run basic tests."""
    logger.info("=== Starting basic test ===")
    
    # Test 1: Python version
    logger.info(f"Python version: {sys.version}")
    
    # Test 2: Current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Test 3: Try to import pandas
    try:
        import pandas as pd
        logger.info(f"Successfully imported pandas version: {pd.__version__}")
    except ImportError as e:
        logger.error(f"Failed to import pandas: {e}")
        return 1
    
    # Test 4: Try to import openpyxl
    try:
        import openpyxl
        logger.info(f"Successfully imported openpyxl version: {openpyxl.__version__}")
    except ImportError as e:
        logger.error(f"Failed to import openpyxl: {e}")
        return 1
    
    # Test 5: Try to create a simple Excel file
    try:
        logger.info("Creating a simple Excel file...")
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        test_file = 'test_excel.xlsx'
        df.to_excel(test_file, index=False)
        logger.info(f"Successfully created test file: {test_file}")
        
        # Verify the file exists
        if os.path.exists(test_file):
            logger.info(f"Verified test file exists: {os.path.abspath(test_file)}")
            # Clean up
            os.remove(test_file)
            logger.info("Cleaned up test file")
        else:
            logger.error("Failed to verify test file creation")
            return 1
            
    except Exception as e:
        logger.error(f"Error creating test Excel file: {e}", exc_info=True)
        return 1
    
    logger.info("=== All tests passed successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
