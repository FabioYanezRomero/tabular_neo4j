"""
Utility functions for CSV file handling.
"""

import os
import pandas as pd
from typing import Tuple, List, Optional
import logging
from Tabular_to_Neo4j.config import CSV_ENCODING, CSV_DELIMITER

def get_primary_entity_from_filename(csv_file_path: str) -> str:
    """
    Extract a potential primary entity name from the CSV filename.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        A string representing a potential entity name (singular form)
    """
    filename = os.path.basename(csv_file_path)
    entity_name = os.path.splitext(filename)[0]
    
    # Basic singularization (very simple approach)
    if entity_name.lower().endswith('s'):
        entity_name = entity_name[:-1]
    
    # Convert snake_case or kebab-case to CamelCase for Neo4j convention
    if '_' in entity_name or '-' in entity_name:
        parts = entity_name.replace('-', '_').split('_')
        entity_name = ''.join(part.capitalize() for part in parts)
    else:
        entity_name = entity_name.capitalize()
    
    return entity_name

def load_csv_safely(file_path: str, header=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Safely load a CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        header: Header parameter for pd.read_csv
        
    Returns:
        Tuple of (DataFrame or None, list of error messages)
    """
    errors = []
    df = None
    
    if not os.path.exists(file_path):
        errors.append(f"File not found: {file_path}")
        return None, errors
    
    try:
        df = pd.read_csv(file_path, header=header, encoding=CSV_ENCODING, delimiter=CSV_DELIMITER)
    except UnicodeDecodeError:
        # Try with different encodings
        encodings = ['latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, header=header, encoding=encoding, delimiter=CSV_DELIMITER)
                break
            except Exception as e:
                continue
        
        if df is None:
            errors.append(f"Failed to decode file with attempted encodings: utf-8, latin1, iso-8859-1, cp1252")
    except Exception as e:
        errors.append(f"Error loading CSV: {str(e)}")
    
    return df, errors

def get_sample_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Get a sample of rows from a DataFrame for LLM analysis.
    
    Args:
        df: The DataFrame to sample from
        n: Number of rows to sample
        
    Returns:
        A DataFrame with the sampled rows
    """
    if len(df) <= n:
        return df
    
    # Take first, last, and some middle rows for a representative sample
    first = df.iloc[:n//3]
    middle = df.iloc[len(df)//2 - n//6:len(df)//2 + n//6]
    last = df.iloc[-n//3:]
    
    return pd.concat([first, middle, last]).head(n)

def df_to_string_sample(df: pd.DataFrame, n: int = 10) -> str:
    """
    Convert a DataFrame to a string representation suitable for LLM prompts.
    
    Args:
        df: The DataFrame to convert
        n: Maximum number of rows to include
        
    Returns:
        A string representation of the DataFrame
    """
    sample = get_sample_rows(df, n)
    return sample.to_string(index=False)
