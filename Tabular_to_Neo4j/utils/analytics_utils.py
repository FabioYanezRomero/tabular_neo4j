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

def detect_patterns(column: pd.Series) -> Dict[str, float]:
    """
    Detect common patterns in the column values.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary mapping pattern names to confidence scores (0-1)
    """
    patterns = {
        'email': 0.0,
        'url': 0.0,
        'phone': 0.0,
        'zipcode': 0.0,
        'id': 0.0,
        'name': 0.0,
        'address': 0.0
    }
    
    # Clean column by dropping NaN values and converting to string
    clean_column = column.dropna().astype(str)
    
    if len(clean_column) == 0:
        return patterns
    
    # Sample values for pattern detection
    sample = clean_column.sample(min(100, len(clean_column)))
    
    # Email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    email_matches = sum(sample.str.match(email_pattern))
    patterns['email'] = email_matches / len(sample) if email_matches > 0 else 0.0
    
    # URL pattern
    url_pattern = r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
    url_matches = sum(sample.str.match(url_pattern))
    patterns['url'] = url_matches / len(sample) if url_matches > 0 else 0.0
    
    # Phone pattern (simple)
    phone_pattern = r'^\+?[\d\s\(\)-]{7,20}$'
    phone_matches = sum(sample.str.match(phone_pattern))
    patterns['phone'] = phone_matches / len(sample) if phone_matches > 0 else 0.0
    
    # Zipcode pattern (simple)
    zipcode_pattern = r'^\d{5}(-\d{4})?$'
    zipcode_matches = sum(sample.str.match(zipcode_pattern))
    patterns['zipcode'] = zipcode_matches / len(sample) if zipcode_matches > 0 else 0.0
    
    # ID pattern (alphanumeric with possible separators)
    id_pattern = r'^[A-Za-z0-9_\-]{3,20}$'
    id_matches = sum(sample.str.match(id_pattern))
    
    # Adjust ID confidence if column name contains 'id' or 'ID'
    column_name = column.name if column.name else ""
    id_name_match = re.search(r'id|ID|Id', str(column_name)) is not None
    
    patterns['id'] = (id_matches / len(sample) + (0.3 if id_name_match else 0)) / (1.3 if id_name_match else 1)
    
    # Name pattern (words with spaces, no numbers)
    name_pattern = r'^[A-Za-z\s\.\-\']{2,50}$'
    name_matches = sum(sample.str.match(name_pattern))
    
    # Adjust name confidence if column name contains 'name' or 'Name'
    name_name_match = re.search(r'name|Name', str(column_name)) is not None
    
    patterns['name'] = (name_matches / len(sample) + (0.3 if name_name_match else 0)) / (1.3 if name_name_match else 1)
    
    # Address pattern (longer strings with numbers and words)
    address_pattern = r'^[\d\s\w\.,#\-\']{10,100}$'
    address_matches = sum(sample.str.match(address_pattern))
    
    # Adjust address confidence if column name contains 'address' or 'Address'
    address_name_match = re.search(r'address|Address', str(column_name)) is not None
    
    patterns['address'] = (address_matches / len(sample) + (0.3 if address_name_match else 0)) / (1.3 if address_name_match else 1)
    
    return patterns

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
        'patterns': detect_patterns(column),
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
