"""Simple test of Python functionality."""
import sys
import os
import platform

def main():
    """Run simple tests."""
    print("=== Python Environment ===")
    print(f"Python Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Working Directory: {os.getcwd()}")
    
    print("\n=== Basic File Operations ===")
    try:
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test content")
        print(f"✓ Created file: {test_file}")
        
        with open(test_file, 'r') as f:
            content = f.read()
        print(f"✓ Read file content: {content}")
        
        os.remove(test_file)
        print(f"✓ Deleted file: {test_file}")
    except Exception as e:
        print(f"✗ File operations failed: {e}")
    
    print("\n=== Testing Imports ===")
    modules = [
        'pandas',
        'openpyxl',
        'numpy',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
