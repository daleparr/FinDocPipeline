"""Check Python environment and imports."""
import sys
import os
import platform

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f" {text} ")
    print("=" * 50)

def main():
    """Run environment checks."""
    # Basic info
    print_header("PYTHON ENVIRONMENT")
    print(f"Python Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check imports
    print_header("IMPORT CHECKS")
    modules = [
        'pandas',
        'openpyxl',
        'numpy',
        'src.etl.parsers.excel_parser'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
    
    # Check file operations
    print_header("FILE OPERATIONS")
    try:
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        print(f"✓ Successfully wrote to {test_file}")
        os.remove(test_file)
        print(f"✓ Successfully deleted {test_file}")
    except Exception as e:
        print(f"✗ File operations failed: {e}")
    
    # Check Excel operations
    print_header("EXCEL OPERATIONS")
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Test"
        ws.append(["Test"])
        excel_file = "test_excel.xlsx"
        wb.save(excel_file)
        print(f"✓ Successfully created {excel_file}")
        
        if os.path.exists(excel_file):
            os.remove(excel_file)
            print(f"✓ Successfully deleted {excel_file}")
        else:
            print(f"✗ {excel_file} was not created")
            
    except Exception as e:
        print(f"✗ Excel operations failed: {e}")
    
    print_header("TEST COMPLETE")

if __name__ == "__main__":
    main()
