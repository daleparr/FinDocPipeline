"""Test script to verify Python environment and basic functionality."""
import sys
import os
import platform
import site

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*40}")
    print(f"{title.upper()}")
    print(f"{'='*40}")

def main():
    """Run environment tests."""
    # Basic Python info
    print_section("Python Information")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Path information
    print_section("Python Path")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")
    
    # Environment variables
    print_section("Environment Variables")
    for var in ['PATH', 'PYTHONPATH', 'PYTHONHOME']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Test basic imports
    print_section("Testing Basic Imports")
    modules = ['pandas', 'openpyxl', 'numpy']
    for mod in modules:
        try:
            __import__(mod)
            print(f"✓ {mod} imported successfully")
        except ImportError as e:
            print(f"✗ {mod} import failed: {e}")
    
    # Test file operations
    print_section("Testing File Operations")
    try:
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        print(f"✓ Successfully wrote to {test_file}")
        os.remove(test_file)
        print(f"✓ Successfully deleted {test_file}")
    except Exception as e:
        print(f"✗ File operations failed: {e}")
    
    print_section("Test Complete")

if __name__ == "__main__":
    main()
