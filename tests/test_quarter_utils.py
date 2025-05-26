""
Tests for quarter_utils module.
"""
import pytest
from pathlib import Path
from src.etl.utils.quarter_utils import normalize_quarter, process_file_path

def test_normalize_quarter():
    ""Test quarter normalization from various filename patterns."""
    # Test patterns from Q1_2025
    assert normalize_quarter("2025fsqtr1rslt.pdf") == "Q1_2025"
    assert normalize_quarter("Q1-2025-report.pdf") == "Q1_2025"
    assert normalize_quarter("2025_q1_results.xlsx") == "Q1_2025"
    
    # Test patterns from Q4_2024
    assert normalize_quarter("4Q24-earnings-press-release.pdf") == "Q4_2024"
    assert normalize_quarter("Q4-2024-Report.pdf") == "Q4_2024"
    assert normalize_quarter("2024_q4_financials.xlsx") == "Q4_2024"
    
    # Test no match
    assert normalize_quarter("annual_report_2023.pdf") is None
    assert normalize_quarter("presentation.pdf") is None

def test_process_file_path(tmp_path):
    ""Test processing file paths with quarter information."""
    # Create test files
    test_files = [
        ("Q1-2025-report.pdf", "Q1_2025"),
        ("4Q24-earnings.pdf", "Q4_2024"),
        ("no_quarter_here.txt", None)
    ]
    
    for filename, expected_quarter in test_files:
        file_path = Path(filename)
        result = process_file_path(file_path, tmp_path)
        
        if expected_quarter:
            expected_path = tmp_path / f"quarter={expected_quarter}" / filename
            assert result == expected_path
            assert result.parent.exists()
        else:
            assert result is None

if __name__ == "__main__":
    pytest.main(["-v", __file__])
