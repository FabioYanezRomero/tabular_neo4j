"""
Utility functions for CSV file handling.
"""
import json
import os
import pandas as pd
import logging
from typing import Tuple, List, Optional, Sequence, Union, Literal
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.config import CSV_ENCODING, CSV_DELIMITER

# Configure logging
logger = get_logger(__name__)

def load_csv_safely(file_path: str, header: Optional[Union[int, Sequence[int], Literal['infer']]] = None) -> Tuple[Optional[pd.DataFrame], Optional[List], Optional[str]]:
    """
    Safely load a CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        header: Header parameter for pd.read_csv
        
    Returns:
        Tuple of (DataFrame or None, potential_header or None, encoding_used or None)
    """
    import chardet
    
    errors = []
    df = None
    potential_header = None
    encoding_used = None
    encoding = CSV_ENCODING  # Default encoding
    confidence = 0.0  # Default confidence
    delimiter = CSV_DELIMITER  # Default delimiter
    
    logger.debug(f"Attempting to load CSV file: {file_path}")
    logger.debug(f"Using configured encoding: {CSV_ENCODING}, delimiter: {CSV_DELIMITER}")
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None, None, None
    
    # Log file size and basic info
    if logger.getEffectiveLevel() == logging.DEBUG: 
        file_size = os.path.getsize(file_path) / 1024  # KB
        logger.debug(f"CSV file size: {file_size:.2f} KB")
    
    try:
        # Detect file encoding
        encoding, confidence = file_encoding_detection(file_path)
        
        # Try different delimiters if the default fails
        delimiter = delimiter_detection(file_path)
        
        # First try to read a few rows to check for header
        temp_df = pd.read_csv(
            file_path, 
            delimiter=delimiter, 
            nrows=5,
            encoding=encoding,
            low_memory=False,
            on_bad_lines='warn'
        )
        
        # Check if first row looks like a header
        first_row = temp_df.iloc[0]
        rest_rows = temp_df.iloc[1:]
        
        # Heuristic: If first row has different data types than the rest,
        # or contains string values while other rows are numeric,
        # it's likely a header
        is_header = False
        
        # Check if first row contains mostly strings while rest are numeric
        if len(temp_df) > 1:
            first_row_str_count = sum(1 for val in first_row if isinstance(val, str))
            rest_rows_str_count = sum(1 for row in rest_rows.itertuples(index=False) 
                                        for val in row if isinstance(val, str))
            
            # If first row has more strings than average of rest rows, likely a header
            if first_row_str_count > rest_rows_str_count / len(rest_rows):
                is_header = True
                potential_header = list(first_row)
                
        # Now read the full file with appropriate header setting
        df = pd.read_csv(
            file_path, 
            delimiter=delimiter, 
            header=0 if is_header else None,
            encoding=encoding,
            low_memory=False,
            on_bad_lines='warn'
        )
            
        # If we didn't detect a header but header=None, the first row is in the data
        # Store it anyway as potential_header for later use
        if not is_header and header is None and len(df) > 0:
            potential_header = list(df.iloc[0])
            
        encoding_used = encoding
        logger.debug(f"Successfully loaded CSV with delimiter: {delimiter}")
    except Exception as e:
        logger.debug(f"Failed to load CSV with delimiter {delimiter}: {str(e)}")
        errors.append(f"Failed with delimiter {delimiter}: {str(e)}")
        
        # If encoding might be the issue, try with utf-8 or latin-1
        df = fallback_encoding(file_path, df, encoding, confidence, delimiter, header)
    
    if df is not None:
        # Check for potential issues
        if len(df.columns) == 1:
            logger.warning("CSV has only one column. This might indicate delimiter issues.")
            errors.append("CSV has only one column. This might indicate delimiter issues.")
            
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logger.debug(f"CSV contains {missing_count} missing values")
    else:
        logger.error("Failed to load CSV file after all attempts")
    
    return df, potential_header, encoding_used


def file_encoding_detection(file_path: str) -> Tuple[str, float]:
    """
    Try to detect the encoding used in the CSV file.
    
    Args:
        file_path: The path to the CSV file
    
    Returns:
        A tuple of (encoding, confidence) where encoding is the detected encoding
        and confidence is the detection confidence score
    """
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read a sample to detect encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        logger.debug(f"Detected encoding: {encoding} with confidence: {confidence}")
    return encoding, confidence


# Make the delimiter detection based on the number of columns obtained with pandas
def delimiter_detection(file_path: str) -> str:
    """
    Try to detect the delimiter used in the CSV file based on the number of columns.
    
    Args:
        file_path: The path to the CSV file
    
    Returns:
        The detected delimiter
    """
    delimiters = [',', ';', '\t', '|']
    best_delim = CSV_DELIMITER
    best_col_count = float('inf')

    for delimiter in delimiters:
        try:
            # Read only a small sample for speed
            df_sample = pd.read_csv(file_path, delimiter=delimiter, header=None, nrows=50, engine="python")
            col_count = len(df_sample.columns)
            # We need more than 1 column, but prefer the delimiter with the FEWEST columns
            if col_count > 1 and col_count < best_col_count:
                best_col_count = col_count
                best_delim = delimiter
        except Exception:
            # Ignore delimiters that break the parser
            continue

    return best_delim
    
    
def fallback_encoding(file_path: str, df: pd.DataFrame, encoding: str, confidence: float, delimiter: str, header: Optional[Union[int, Sequence[int], Literal['infer']]]) -> pd.DataFrame:
    """
    Try to load a DataFrame with a fallback encoding if the default fails.
    
    Args:
        file_path: The path to the CSV file
        df: The DataFrame to load (initial attempt)
        encoding: The encoding to try first
        confidence: Confidence score from chardet
        delimiter: The delimiter used in the CSV file
        header: The header argument for pd.read_csv

    Returns:
        The loaded DataFrame
    """
    # Retry with common encodings whenever the initial attempt failed (df is None)
    # or when the detected encoding has low confidence. This covers cases where
    # chardet claims high confidence for "ascii" but the file actually contains
    # a few non-ASCII bytes further down the file (e.g. movie titles with accents).
    if df is None or confidence < 0.9:
        fallback_encodings = ['utf-8', 'latin-1', 'cp1252']
        for fb_encoding in fallback_encodings:
            if fb_encoding != encoding:
                try:
                    df = pd.read_csv(file_path, 
                    delimiter=delimiter, 
                    header=header, 
                    encoding=fb_encoding, 
                    low_memory=False, 
                    on_bad_lines='warn'
                    )
                    logger.debug(f"Successfully loaded CSV with delimiter: {delimiter} and encoding: {fb_encoding}")
                    return df  # Return the DataFrame immediately upon success
                except Exception as e:
                    logger.debug(f"Failed with delimiter {delimiter} and encoding {fb_encoding}: {str(e)}")
    return df



def get_sample_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if len(df) <= n:
        return df
    n_each_third = max(1, n // 3) # Ensure at least 1 row from each section if n is small
    n_middle_half = max(1, n // 6)

    first_count = n_each_third
    last_count = n_each_third
    middle_count = n - first_count - last_count

    if middle_count <= 0: # Adjust if n is too small for 3 parts
        first_count = (n + 1) // 2
        last_count = n // 2
        middle_count = 0
        middle = pd.DataFrame()
    
    first = df.iloc[:first_count]
    
    if middle_count > 0:
        middle_start_idx = max(0, len(df)//2 - middle_count//2)
        middle_end_idx = middle_start_idx + middle_count
        middle = df.iloc[middle_start_idx:middle_end_idx]
    
    last = df.iloc[-last_count:]
    
    # Make sure we don't have too many rows due to small df or rounding
    # and also handle potential overlaps if df is very small
    combined = pd.concat([first, middle, last]).drop_duplicates().head(n)
    return combined


def df_to_json_sample(df: pd.DataFrame, n: int = 10) -> str:
    """
    Convert a DataFrame sample to a JSON string (list of dictionaries).
    Each dictionary represents a row with column names as keys.
    Returns a formatted string with proper indentation and line breaks.
    """
    sample_df = get_sample_rows(df, n)
    # Convert to list of dictionaries with column names as keys
    records = sample_df.to_dict(orient='records')
    
    # Format with indentation and line breaks for better readability
    formatted_json = json.dumps(records, indent=2)
    
    # For prompt display, we want each row on a separate line
    # This makes a more readable format for the LLM
    return formatted_json


# Function removed as per user request
