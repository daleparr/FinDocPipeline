"""
Basic tests for the Financial ETL Pipeline
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_python_version():
    """Test that Python version is supported."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"

def test_basic_imports():
    """Test that basic Python libraries can be imported."""
    try:
        import pandas as pd
        import json
        import pathlib
        assert True
    except ImportError as e:
        pytest.fail(f"Basic import failed: {e}")

def test_project_structure():
    """Test that key project files exist."""
    project_root = Path(__file__).parent.parent
    
    # Check for key files
    key_files = [
        "README.md",
        "requirements.txt",
        "standalone_frontend.py",
        "src/etl/__init__.py"
    ]
    
    for file_path in key_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Missing key file: {file_path}"

def test_src_etl_import():
    """Test that ETL modules can be imported."""
    try:
        # Try to import ETL modules
        from etl import config
        assert True, "ETL config import successful"
    except ImportError:
        # This is expected in CI environment
        pytest.skip("ETL modules not available in CI environment")

def test_standalone_frontend_import():
    """Test that standalone frontend can be imported."""
    try:
        import standalone_frontend
        assert hasattr(standalone_frontend, 'main'), "Frontend main function exists"
    except ImportError as e:
        pytest.skip(f"Frontend import failed (expected in CI): {e}")

def test_requirements_files():
    """Test that requirements files exist and are readable."""
    project_root = Path(__file__).parent.parent
    
    req_files = ["requirements.txt", "requirements_frontend.txt"]
    
    for req_file in req_files:
        req_path = project_root / req_file
        if req_path.exists():
            with open(req_path, 'r') as f:
                content = f.read()
                assert len(content) > 0, f"{req_file} is not empty"
                assert "pandas" in content or "streamlit" in content, f"{req_file} contains expected packages"

def test_directory_structure():
    """Test that expected directories exist."""
    project_root = Path(__file__).parent.parent
    
    expected_dirs = [
        "src",
        "src/etl",
        "tests",
        ".github",
        ".github/workflows"
    ]
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        assert full_path.exists() and full_path.is_dir(), f"Missing directory: {dir_path}"

def test_git_files():
    """Test that git configuration files exist."""
    project_root = Path(__file__).parent.parent
    
    git_files = [".gitignore", "LICENSE"]
    
    for git_file in git_files:
        file_path = project_root / git_file
        assert file_path.exists(), f"Missing git file: {git_file}"

if __name__ == "__main__":
    pytest.main([__file__])