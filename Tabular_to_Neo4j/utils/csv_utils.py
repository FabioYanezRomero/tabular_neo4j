"""
Utility functions for CSV file handling.
"""

import os
import pandas as pd
from typing import Tuple, List, Optional
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.config import CSV_ENCODING, CSV_DELIMITER

# Configure logging
logger = get_logger(__name__)

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
    
    logger.debug(f"Attempting to load CSV file: {file_path}")
    logger.debug(f"Using configured encoding: {CSV_ENCODING}, delimiter: {CSV_DELIMITER}")
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None, errors
    
    # Log file size and basic info
    file_size = os.path.getsize(file_path) / 1024  # KB
    logger.debug(f"CSV file size: {file_size:.2f} KB")
    
    # Try to detect file type based on extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.csv':
        logger.warning(f"File has non-CSV extension: {ext}. Will attempt to load anyway.")
    
    try:
        logger.debug(f"Attempting to load CSV with primary encoding: {CSV_ENCODING}")
        df = pd.read_csv(file_path, header=header, encoding=CSV_ENCODING, delimiter=CSV_DELIMITER)
        logger.info(f"Successfully loaded CSV file with primary encoding: {CSV_ENCODING}")
    except UnicodeDecodeError as e:
        logger.warning(f"Unicode decode error with {CSV_ENCODING}: {str(e)}")
        logger.info("Attempting to load with alternative encodings")
        
        # Try with different encodings
        encodings = ['latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                logger.debug(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, header=header, encoding=encoding, delimiter=CSV_DELIMITER)
                logger.info(f"Successfully loaded CSV file with encoding: {encoding}")
                break
            except Exception as e:
                logger.warning(f"Failed to load CSV with encoding {encoding}: {str(e)}")
                continue
        
        if df is None:
            error_msg = f"Failed to decode file with attempted encodings: {CSV_ENCODING}, latin1, iso-8859-1, cp1252"
            logger.error(error_msg)
            errors.append(error_msg)
    except pd.errors.EmptyDataError as e:
        error_msg = f"CSV file is empty: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
    except pd.errors.ParserError as e:
        error_msg = f"CSV parsing error (malformed CSV): {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        
        # Try with different delimiters as fallback
        if CSV_DELIMITER == ',':
            alt_delimiters = [';', '\t', '|']
            logger.info(f"Attempting to load with alternative delimiters")
            
            for delimiter in alt_delimiters:
                try:
                    logger.debug(f"Trying delimiter: '{delimiter}'")
                    df = pd.read_csv(file_path, header=header, encoding=CSV_ENCODING, delimiter=delimiter)
                    logger.info(f"Successfully loaded CSV file with delimiter: '{delimiter}'")
                    break
                except Exception as e:
                    logger.debug(f"Failed with delimiter '{delimiter}': {str(e)}")
                    continue
    except Exception as e:
        error_msg = f"Error loading CSV: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    # Log dataframe info if successfully loaded
    if df is not None:
        logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
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
