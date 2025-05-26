"""
Test script for file discovery and parsing functionality.
"""
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent.resolve()
        sys.path.insert(0, str(project_root))
        
        print("=" * 80)
        print("TESTING FLEXIBLE FILE DISCOVERY AND PARSING")
        print("=" * 80)
        
        # Import the file discovery module
        from src.etl.file_discovery import discover_and_process, discover_files
        
        # Test with the actual data_sources directory
        raw_data_dir = project_root / "data_sources"
        
        if not raw_data_dir.exists():
            print(f"Error: Data sources directory not found at {raw_data_dir}")
            print("Please ensure you have the correct directory structure with test data.")
            return 1
            
        print(f"\n1. DISCOVERING ALL FILES")
        print("-" * 80)
        
        # First, just discover all files without processing
        files = discover_files(raw_data_dir)
        print(f"Found {len(files)} files:")
        
        # Print the discovered files with their inferred metadata
        for bank, quarter, file_path, doc_type in files:
            print(f"Bank: {bank:15} | Quarter: {quarter:10} | Type: {doc_type:12} | File: {file_path.name}")
            
        print("\n\n2. PROCESSING ALL FILES")
        print("-" * 80)
        
        # Process all files with the default bank name
        discover_and_process(raw_data_dir, default_bank="Citigroup")
        
        print("\n" + "=" * 80)
        print("FILE DISCOVERY AND PARSING TEST COMPLETED")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
