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


def calculate_value_length_stats(column: pd.Series) -> Dict[str, float]:
    """
    Calculate statistics about the length of string values in a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with length statistics (min, max, mean, median)
    """
    # Convert to string and calculate lengths
    clean_column = column.dropna()
    if len(clean_column) == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
    
    # Convert all values to strings and get their lengths
    try:
        lengths = clean_column.astype(str).apply(len)
        return {
            'min': float(lengths.min()),
            'max': float(lengths.max()),
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'std': float(lengths.std()) if len(lengths) > 1 else 0.0
        }
    except Exception as e:
        # Return default values if calculation fails
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}

def detect_patterns(column: pd.Series) -> Dict[str, float]:
    """
    Detect common patterns in the column values that might indicate specific entity types.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with pattern match ratios
    """
    clean_column = column.dropna().astype(str)
    if len(clean_column) == 0:
        return {}
    
    # Sample values to avoid processing too many rows
    sample_size = min(1000, len(clean_column))
    sample = clean_column.sample(sample_size) if len(clean_column) > sample_size else clean_column
    
    patterns = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone': r'^\+?[0-9\-\(\)\s]{7,20}$',
        'url': r'^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$',
        'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        'postal_code': r'^\d{5}(-\d{4})?$|^[A-Z]\d[A-Z]\s?\d[A-Z]\d$',  # US and Canadian postal codes
        'credit_card': r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$',
        'alphanumeric_id': r'^[A-Za-z0-9-_]{3,}$',
        'numeric_id': r'^\d+$'
    }
    
    results = {}
    for pattern_name, regex in patterns.items():
        match_count = sum(1 for val in sample if re.match(regex, val))
        match_ratio = match_count / len(sample)
        if match_ratio > 0.5:  # Only include significant patterns
            results[pattern_name] = match_ratio
    
    return results

def analyze_value_distribution(column: pd.Series) -> Dict[str, Any]:
    """
    Analyze the distribution of values in a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with distribution metrics
    """
    clean_column = column.dropna()
    if len(clean_column) == 0 or clean_column.nunique() <= 1:
        return {'uniformity': 0.0, 'top_value_frequency': 0.0}
    
    # Calculate value frequencies
    value_counts = clean_column.value_counts(normalize=True)
    
    # Uniformity: how evenly distributed the values are (higher = more uniform)
    # Using a simplified entropy calculation
    uniformity = 1.0 - value_counts.max()
    
    # Top value frequency
    top_value_frequency = value_counts.iloc[0] if len(value_counts) > 0 else 0.0
    
    return {
        'uniformity': float(uniformity),
        'top_value_frequency': float(top_value_frequency),
        'unique_count': int(clean_column.nunique())
    }

def analyze_column(column: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with analysis results
    """
    # Basic analytics
    uniqueness = calculate_uniqueness_ratio(column)
    cardinality = calculate_cardinality(column)
    data_type = detect_data_type(column)
    missing_percentage = calculate_missing_percentage(column)
    
    # Advanced analytics
    value_lengths = calculate_value_length_stats(column)
    patterns = detect_patterns(column)
    distribution = analyze_value_distribution(column)
    
    # Sample values for reference
    sample_values = column.dropna().sample(min(5, len(column.dropna()))).tolist() if len(column.dropna()) > 0 else []
    
    # Combine all analytics
    analysis = {
        'column_name': column.name,
        'uniqueness': uniqueness,
        'cardinality': cardinality,
        'data_type': data_type,
        'missing_percentage': missing_percentage,
        'value_lengths': value_lengths,
        'patterns': patterns,
        'distribution': distribution,
        'sample_values': sample_values
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
