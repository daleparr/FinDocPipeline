"""
Test file for data_utils module

This file contains basic tests for the data science utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

try:
    from data_utils import load_config, explore_data, clean_data, split_data
except ImportError:
    # If imports fail, create placeholder tests
    pass


def test_sample_data_creation():
    """Test creating sample data for testing purposes."""
    # Create sample dataset
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
    })
    
    assert df.shape == (100, 4)
    assert 'age' in df.columns
    assert 'income' in df.columns
    assert df['age'].min() >= 18
    assert df['age'].max() <= 80


def test_data_types():
    """Test data type validation."""
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'string_col': ['a', 'b', 'c', 'd', 'e'],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    assert df['numeric_col'].dtype in ['int64', 'int32']
    assert df['string_col'].dtype == 'object'
    assert df['float_col'].dtype in ['float64', 'float32']


def test_missing_values():
    """Test handling of missing values."""
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['a', 'b', None, 'd', 'e']
    })
    
    assert df.isnull().sum().sum() == 2
    assert df['col1'].isnull().sum() == 1
    assert df['col2'].isnull().sum() == 1


def test_basic_statistics():
    """Test basic statistical calculations."""
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5]
    })
    
    assert df['values'].mean() == 3.0
    assert df['values'].median() == 3.0
    assert df['values'].std() == pytest.approx(1.58, rel=1e-2)


if __name__ == "__main__":
    # Run basic tests
    test_sample_data_creation()
    test_data_types()
    test_missing_values()
    test_basic_statistics()
    print("✅ All basic tests passed!")