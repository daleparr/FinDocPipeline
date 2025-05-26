"""
Test script for the Excel parser.

This script demonstrates how to use the ExcelParser class to parse an Excel file
and inspect its contents.
"""
import sys
import os
import json
import logging
import traceback
from pathlib import Path

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for normal operation
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
try:
    import pandas as pd
    from openpyxl import Workbook
    from src.etl.parsers.excel_parser import ExcelParser
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install the required packages with: pip install pandas openpyxl")
    sys.exit(1)

def create_test_excel_file(file_path: Path) -> bool:
    """Create a test Excel file with sample data.
    
    Args:
        file_path: Path where the Excel file should be created.
        
    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    try:
        import pandas as pd
        from io import BytesIO
        
        logger.info("Creating test Excel file...")
        
        # Create a sample Excel file in memory
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Financial metrics
            df1 = pd.DataFrame({
                'Metric': ['Revenue', 'Expenses', 'Net Income', 'EPS'],
                'Q1_2025': [1000, 700, 300, 1.50],
                'Q4_2024': [950, 650, 300, 1.50],
                'Q3_2024': [900, 600, 300, 1.50]
            })
            df1.to_excel(writer, sheet_name='Financials', index=False)
            
            # Sheet 2: Risk metrics
            df2 = pd.DataFrame({
                'Risk Category': ['Credit', 'Market', 'Operational', 'Liquidity'],
                'Current': [3.2, 2.8, 2.5, 4.1],
                'Previous': [3.0, 2.7, 2.4, 4.0],
                'Change': [0.2, 0.1, 0.1, 0.1],
                'Threshold': [5.0, 5.0, 5.0, 5.0]
            })
            df2.to_excel(writer, sheet_name='Risk Metrics', index=False)
        
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the Excel file to disk
        with open(file_path, 'wb') as f:
            f.write(output.getvalue())
        
        logger.info(f"Successfully created test Excel file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create test Excel file: {e}", exc_info=True)
        return False

def create_test_excel_file_simple(file_path: Path) -> bool:
    """Create a simple test Excel file using openpyxl directly."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        
        # Create a new workbook and select the active worksheet
        wb = Workbook()
        
        # First sheet: Financials
        ws1 = wb.active
        ws1.title = "Financials"
        
        # Add headers
        headers = ['Metric', 'Q1_2025', 'Q4_2024', 'Q3_2024']
        ws1.append(headers)
        
        # Add data
        data = [
            ['Revenue', 1000, 950, 900],
            ['Expenses', 700, 650, 600],
            ['Net Income', 300, 300, 300],
            ['EPS', 1.50, 1.50, 1.50]
        ]
        for row in data:
            ws1.append(row)
        
        # Add some formatting
        for cell in ws1[1]:  # Header row
            cell.font = Font(bold=True)
        
        # Second sheet: Risk Metrics
        ws2 = wb.create_sheet("Risk Metrics")
        
        # Add headers
        headers = ['Risk Category', 'Current', 'Previous', 'Change', 'Threshold']
        ws2.append(headers)
        
        # Add data
        data = [
            ['Credit', 3.2, 3.0, 0.2, 5.0],
            ['Market', 2.8, 2.7, 0.1, 5.0],
            ['Operational', 2.5, 2.4, 0.1, 5.0],
            ['Liquidity', 4.1, 4.0, 0.1, 5.0]
        ]
        for row in data:
            ws2.append(row)
        
        # Add some formatting
        for cell in ws2[1]:  # Header row
            cell.font = Font(bold=True)
        
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        wb.save(file_path)
        logger.info(f"Created test Excel file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating test Excel file: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Test the Excel parser with a sample file."""
    try:
        # Define the test file path
        test_file = project_root / 'test_financials_q1_2025.xlsx'
        
        # Create the test Excel file using the simple method
        logger.info(f"Creating test Excel file: {test_file}")
        if not create_test_excel_file_simple(test_file):
            logger.error("Failed to create test Excel file. Exiting.")
            return 1
        
        # Verify the file was created
        if not test_file.exists():
            logger.error(f"Test file was not created: {test_file}")
            return 1
            
        logger.info(f"Test file created successfully: {test_file}")
        
        # Create the parser
        logger.info("Creating ExcelParser instance...")
        parser = ExcelParser(bank='test_bank', quarter='Q1_2025')
        
        # Parse the Excel file
        logger.info(f"Parsing Excel file: {test_file}")
        try:
            result = parser.parse(test_file)
            logger.info("Successfully parsed Excel file")
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        # Print basic results
        print("\n=== Parser Results ===")
        print(f"Bank: {result.get('bank', 'N/A')}")
        print(f"Quarter: {result.get('quarter', 'N/A')}")
        print(f"Document Type: {result.get('document_type', 'N/A')}")
        print(f"File Path: {result.get('file_path', 'N/A')}")
        
        # Print sheet information
        if 'content' in result and 'tables' in result['content']:
            print("\n=== Sheets ===")
            for sheet_name, sheet_data in result['content']['tables'].items():
                print(f"\nSheet: {sheet_name}")
                print(f"  Rows: {sheet_data.get('shape', {}).get('rows', 'N/A')}, "
                      f"Columns: {sheet_data.get('shape', {}).get('columns', 'N/A')}")
                
                # Print columns
                if 'columns' in sheet_data:
                    print(f"  Columns: {[col.get('name', 'N/A') for col in sheet_data['columns']]}")
                
                # Print first few rows of data
                if 'data' in sheet_data and sheet_data['data']:
                    print("  Sample Data (first 3 rows):")
                    for i, row in enumerate(sheet_data['data'][:3]):
                        print(f"    Row {i+1}: {row}")
        
        # Save the full result to a JSON file for inspection
        output_file = test_file.with_suffix('.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nFull output saved to: {output_file}")
            
            # Validate the JSON output
            with open(output_file, 'r', encoding='utf-8') as f:
                json.load(f)  # Validate JSON is valid
            print("Output JSON is valid")
            
        except Exception as e:
            logger.error(f"Failed to save or validate JSON output: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        print("\nTest completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main()
