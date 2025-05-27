"""
Data Science Utility Functions

This module contains common utility functions for data science projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, **kwargs)
    elif file_path.suffix.lower() == '.json':
        df = pd.read_json(file_path, **kwargs)
    elif file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Data loaded from {file_path}. Shape: {df.shape}")
    return df


def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save data to various file formats.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        **kwargs: Additional arguments for pandas save functions
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix.lower() == '.csv':
        df.to_csv(file_path, index=False, **kwargs)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df.to_excel(file_path, index=False, **kwargs)
    elif file_path.suffix.lower() == '.json':
        df.to_json(file_path, **kwargs)
    elif file_path.suffix.lower() == '.parquet':
        df.to_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Data saved to {file_path}")


def explore_data(df: pd.DataFrame) -> None:
    """
    Perform basic exploratory data analysis.
    
    Args:
        df: DataFrame to explore
    """
    print("=" * 50)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 50)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn Information:")
    print(df.info())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\nNumerical Columns Summary:")
    print(df.describe())
    
    print("\nCategorical Columns Summary:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())


def clean_data(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Clean data based on configuration.
    
    Args:
        df: DataFrame to clean
        config: Configuration dictionary
    
    Returns:
        pandas.DataFrame: Cleaned data
    """
    df_clean = df.copy()
    
    if config is None:
        config = {}
    
    # Handle missing values
    missing_strategy = config.get('features', {}).get('handle_missing', 'median')
    
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    if missing_strategy == 'median':
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())
    elif missing_strategy == 'mean':
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].mean())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    logger.info("Data cleaning completed")
    return df_clean


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame to split
        target_column: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    """
    Plot correlation matrix for numerical columns.
    
    Args:
        df: DataFrame to analyze
        figsize: Figure size for the plot
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation matrix")
        return
    
    plt.figure(figsize=figsize)
    correlation_matrix = df[numerical_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_missing_values(df: pd.DataFrame, figsize: tuple = (12, 6)) -> None:
    """
    Plot missing values in the dataset.
    
    Args:
        df: DataFrame to analyze
        figsize: Figure size for the plot
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values found in the dataset")
        return
    
    plt.figure(figsize=figsize)
    missing.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Data Science Utilities Module")
    print("Import this module to use the utility functions")