"""
Utility functions for column analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import re
from datetime import datetime

def calculate_uniqueness_ratio(column: pd.Series) -> float:
    """
    Calculate the uniqueness ratio of a column (distinct values / total values).
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Uniqueness ratio as a float between 0 and 1
    """
    if len(column) == 0:
        return 0.0
    return column.nunique() / len(column)

def calculate_cardinality(column: pd.Series) -> int:
    """
    Calculate the cardinality (number of distinct values) of a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Cardinality as an integer
    """
    return column.nunique()

def calculate_missing_percentage(column: pd.Series) -> float:
    """
    Calculate the percentage of missing values in a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Missing value percentage as a float between 0 and 1
    """
    if len(column) == 0:
        return 0.0
    return column.isna().mean()

def detect_data_type(column: pd.Series) -> str:
    """
    Detect the predominant data type of a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        String representing the data type ('numeric', 'string', 'date', 'boolean', 'mixed')
    """
    # Clean column by dropping NaN values
    clean_column = column.dropna()
    
    if len(clean_column) == 0:
        return 'unknown'
    
    # Check if all values are numeric
    try:
        pd.to_numeric(clean_column)
        return 'numeric'
    except:
        pass
    
    # Check if all values are boolean
    if set(clean_column.astype(str).str.lower().unique()).issubset({'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}):
        return 'boolean'
    
    # Check if all values are dates
    date_count = 0
    for val in clean_column.astype(str).sample(min(100, len(clean_column))):
        try:
            datetime.strptime(val, '%Y-%m-%d')
            date_count += 1
        except:
            try:
                datetime.strptime(val, '%d/%m/%Y')
                date_count += 1
            except:
                try:
                    datetime.strptime(val, '%m/%d/%Y')
                    date_count += 1
                except:
                    pass
    
    if date_count / len(clean_column.sample(min(100, len(clean_column)))) > 0.8:
        return 'date'
    
    # Default to string
    return 'string'


def analyze_column(column: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'column_name': column.name,
        'uniqueness_ratio': calculate_uniqueness_ratio(column),
        'cardinality': calculate_cardinality(column),
        'data_type': detect_data_type(column),
        'missing_percentage': calculate_missing_percentage(column),
        'sample_values': column.dropna().sample(min(5, len(column.dropna()))).tolist() if len(column.dropna()) > 0 else []
    }
    return analysis

def analyze_all_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all columns in a DataFrame.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary mapping column names to their analysis results
    """
    results = {}
    
    for column_name in df.columns:
        results[column_name] = analyze_column(df[column_name])
    
    return results
