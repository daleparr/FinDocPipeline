"""Check the project structure and module imports."""
import os
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f" {text} ")
    print("=" * 50)

def main():
    """Check project structure and imports."""
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'
    
    print_header("PROJECT STRUCTURE")
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    
    # Check if src is a Python package
    print("\nChecking src directory:")
    if not (src_dir / '__init__.py').exists():
        print("⚠ src/__init__.py not found. Creating it...")
        try:
            (src_dir / '__init__.py').touch()
            print("✓ Created src/__init__.py")
        except Exception as e:
            print(f"✗ Failed to create src/__init__.py: {e}")
    else:
        print("✓ Found src/__init__.py")
    
    # Check etl package
    etl_dir = src_dir / 'etl'
    print("\nChecking etl package:")
    if not etl_dir.exists():
        print("✗ etl directory not found")
        return 1
    
    if not (etl_dir / '__init__.py').exists():
        print("⚠ etl/__init__.py not found. Creating it...")
        try:
            (etl_dir / '__init__.py').touch()
            print("✓ Created etl/__init__.py")
        except Exception as e:
            print(f"✗ Failed to create etl/__init__.py: {e}")
    else:
        print("✓ Found etl/__init__.py")
    
    # Check parsers package
    parsers_dir = etl_dir / 'parsers'
    print("\nChecking parsers package:")
    if not parsers_dir.exists():
        print("✗ parsers directory not found")
        return 1
    
    if not (parsers_dir / '__init__.py').exists():
        print("⚠ parsers/__init__.py not found. Creating it...")
        try:
            (parsers_dir / '__init__.py').touch()
            print("✓ Created parsers/__init__.py")
        except Exception as e:
            print(f"✗ Failed to create parsers/__init__.py: {e}")
    else:
        print("✓ Found parsers/__init__.py")
    
    # Check for required files
    required_files = [
        (parsers_dir / 'base_parser.py', 'Base Parser'),
        (parsers_dir / 'excel_parser.py', 'Excel Parser'),
        (parsers_dir / '__init__.py', 'Parsers Package')
    ]
    
    print("\nChecking for required files:")
    all_files_exist = True
    for file, name in required_files:
        if file.exists():
            print(f"✓ Found {name}: {file}")
        else:
            print(f"✗ Missing {name}: {file}")
            all_files_exist = False
    
    if not all_files_exist:
        return 1
    
    # Test importing the module
    print_header("TESTING IMPORTS")
    try:
        # Add src to Python path
        if str(src_dir.parent) not in sys.path:
            sys.path.insert(0, str(src_dir.parent))
        
        print("\nTrying to import ExcelParser...")
        from src.etl.parsers.excel_parser import ExcelParser
        print("✓ Successfully imported ExcelParser")
        
        print("\nCreating ExcelParser instance...")
        parser = ExcelParser(bank="test_bank", quarter="Q1_2025")
        print("✓ Successfully created ExcelParser instance")
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✓ Project structure and imports look good!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
