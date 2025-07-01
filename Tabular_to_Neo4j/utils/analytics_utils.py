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
    Detect the predominant data type of a column with more precise categorization.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        String representing the data type ('integer', 'float', 'string', 'date', 'datetime', 'boolean', 'categorical', 'unknown')
    """
    # Clean column by dropping NaN values
    clean_column = column.dropna()
    
    if len(clean_column) == 0:
        return 'unknown'
    
    # Check original pandas dtype first
    orig_dtype = str(column.dtype)
    if 'int' in orig_dtype:
        return 'integer'
    elif 'float' in orig_dtype:
        return 'float'
    elif 'datetime' in orig_dtype:
        return 'datetime'
    elif 'bool' in orig_dtype:
        return 'boolean'
    elif 'category' in orig_dtype:
        return 'categorical'
    
    # Try to convert to numeric
    try:
        numeric_values = pd.to_numeric(clean_column)
        # Determine if integer or float
        if all(numeric_values == numeric_values.astype(int)):
            return 'integer'
        else:
            return 'float'
    except:
        pass
    
    # Check if all values are boolean
    if set(clean_column.astype(str).str.lower().unique()).issubset({'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}):
        return 'boolean'
    
    # Check if values are categorical (few unique values compared to total)
    unique_ratio = len(clean_column.unique()) / len(clean_column)
    if unique_ratio < 0.1 and len(clean_column.unique()) < 20:
        return 'categorical'
    
    # Check if all values are dates
    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']
    date_count = 0
    datetime_count = 0
    sample_size = min(100, len(clean_column))
    sample = clean_column.astype(str).sample(sample_size)
    
    for val in sample:
        # Check for date formats
        for date_format in date_formats:
            try:
                datetime.strptime(val, date_format)
                date_count += 1
                break
            except:
                pass
        
        # Check for datetime formats
        try:
            # Try common datetime formats
            if ' ' in val and ((':' in val) or ('T' in val)):
                pd.to_datetime(val)
                datetime_count += 1
        except:
            pass
    
    if date_count / sample_size > 0.8:
        return 'date'
    elif datetime_count / sample_size > 0.8:
        return 'datetime'
    
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
    Perform comprehensive analysis on a column, including DeepJoin/PLM-compatible contextualization statistics and adaptive sampling.
    Args:
        column: Pandas Series representing a column
    Returns:
        Dictionary with analysis results
    """
    uniqueness_ratio = calculate_uniqueness_ratio(column)
    cardinality = calculate_cardinality(column)
    data_type = detect_data_type(column)
    missing_percentage = calculate_missing_percentage(column)
    value_lengths = calculate_value_length_stats(column)
    distribution = analyze_value_distribution(column)
    total_count = len(column)
    mode_series = column.mode(dropna=True)
    mode_values = mode_series.tolist()[:5] if not mode_series.empty else []
    min_value = max_value = avg_value = ''
    quantiles = {}
    sampled_values = []
    contextual_description = ''
    if uniqueness_ratio < 0.05:
        cardinality_type = 'low'
    elif uniqueness_ratio < 0.5:
        cardinality_type = 'medium'
    else:
        cardinality_type = 'high'

    # Numerical columns
    if data_type in ['integer', 'float']:
        try:
            min_value = float(column.min()) if not column.dropna().empty else ''
            max_value = float(column.max()) if not column.dropna().empty else ''
            avg_value = float(column.mean()) if not column.dropna().empty else ''
            quantiles = column.quantile([0.25, 0.5, 0.75]).to_dict() if not column.dropna().empty else {}
        except Exception:
            min_value = max_value = avg_value = ''
            quantiles = {}
        # Distribution descriptor (skew, kurtosis, etc.)
        desc = []
        if not column.dropna().empty:
            try:
                skew = float(column.skew())
                if skew > 1:
                    desc.append('skewed right')
                elif skew < -1:
                    desc.append('skewed left')
                else:
                    desc.append('symmetric')
            except Exception:
                pass
            try:
                kurt = float(column.kurtosis())
                if kurt > 3:
                    desc.append('heavy tails')
                elif kurt < 3:
                    desc.append('light tails')
            except Exception:
                pass
        contextual_description = ', '.join(desc)
        if cardinality > 10000:
            sampled_values = []  # No raw samples
        else:
            sampled_values = mode_values[:5] if cardinality < 100 else mode_values[:3]
    # Temporal columns
    elif data_type in ['date', 'datetime']:
        try:
            min_value = str(column.min()) if not column.dropna().empty else ''
            max_value = str(column.max()) if not column.dropna().empty else ''
            avg_value = ''
        except Exception:
            min_value = max_value = ''
        contextual_description = f"start: {min_value}, end: {max_value}"
        sampled_values = mode_values[:3]
    
    # Categorical columns
    elif data_type == 'categorical':
        sampled_values = mode_values[:5] if cardinality < 100 else mode_values[:3]
        contextual_description = f"{cardinality} unique"
    
    # Text/multi-value columns
    elif data_type == 'string':
        if cardinality > 10000:
            sampled_values = []
            contextual_description = f"{cardinality} unique, text format"
        elif cardinality > 1000:
            sampled_values = mode_values[:3]
            contextual_description = f"{cardinality} unique, text format"
        else:
            # Truncate to 512 tokens or 5 samples
            samples = column.dropna().astype(str).unique().tolist()
            sampled_values = samples[:5]
            contextual_description = f"{cardinality} unique, sample values"
    else:
        sampled_values = mode_values[:3]
        contextual_description = f"{cardinality} unique"

    # Flag for low-cardinality numeric columns that are effectively categorical
    # Use a percentage threshold for 'effectively categorical' numeric columns
    is_effectively_categorical = False
    EFFECTIVE_CATEGORICAL_PERCENT = 0.05  # 5% unique values or fewer is categorical
    # Only check if total_count > 0 to avoid division by zero
    if data_type in ['integer', 'float'] and total_count > 0:
        if (cardinality / total_count) <= EFFECTIVE_CATEGORICAL_PERCENT:
            is_effectively_categorical = True

    analysis = {
        'column_name': column.name,
        'original_column_name': column.name,
        'uniqueness_ratio': uniqueness_ratio,
        'cardinality': cardinality,
        'data_type': data_type,
        'missing_percentage': missing_percentage,
        'value_lengths': value_lengths,
        'distribution': distribution,
        'total_count': total_count,
        'mode_values': mode_values,
        'min_value': min_value,
        'max_value': max_value,
        'avg_value': avg_value,
        'quantiles': quantiles,
        'sampled_values': sampled_values,
        'contextual_description': contextual_description,
        'cardinality_type': cardinality_type,
        'is_effectively_categorical': is_effectively_categorical,
    }
    return analysis


def analyze_all_columns(df: pd.DataFrame, llm_to_original: dict = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all columns in a DataFrame, optionally referencing original column names.
    Args:
        df: Pandas DataFrame
        llm_to_original: Optional dict mapping LLM-inferred column names to original names
    Returns:
        Dictionary mapping LLM-inferred column names to their analysis results (with both names)
    """
    results = {}
    for column_name in df.columns:
        original_name = llm_to_original[column_name] if llm_to_original and column_name in llm_to_original else column_name
        # Run analytics on the original column if available, else fallback
        if original_name in df.columns:
            col_series = df[original_name]
        else:
            col_series = df[column_name]
        analysis = analyze_column(col_series)
        analysis['column_name'] = column_name
        analysis['original_column_name'] = original_name
        results[column_name] = analysis
    return results
