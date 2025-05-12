"""
Input nodes for the LangGraph CSV analysis pipeline.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import load_csv_safely
from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_csv_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Load a CSV file into a pandas DataFrame without assuming headers.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with raw_dataframe and potential header
    """
    # Initialize error messages list if not present
    if 'error_messages' not in state:
        state['error_messages'] = []
    
    csv_path = state['csv_file_path']
    logger.info(f"Loading CSV file: {csv_path}")
    
    # Check if file exists
    import os
    if not os.path.exists(csv_path):
        error_msg = f"CSV file not found: {csv_path}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        state['raw_dataframe'] = None
        state['header_row_if_present'] = []
        return state
    
    # Log file size
    file_size = os.path.getsize(csv_path) / 1024  # KB
    logger.debug(f"CSV file size: {file_size:.2f} KB")
    
    # Load the CSV file
    logger.debug(f"Attempting to load CSV file with automatic encoding detection")
    df, errors = load_csv_safely(csv_path, header=None)
    
    # Add any errors to the state
    if errors:
        for error in errors:
            logger.warning(f"CSV loading issue: {error}")
        state['error_messages'].extend(errors)
        
    if df is not None:
        # Log basic dataframe info
        row_count = len(df)
        col_count = len(df.columns)
        logger.info(f"Successfully loaded CSV with {row_count} rows and {col_count} columns")
        logger.debug(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Check for missing values
        missing_values = df.isna().sum().sum()
        if missing_values > 0:
            logger.warning(f"CSV contains {missing_values} missing values")
            state['error_messages'].append(f"CSV contains {missing_values} missing values")
        
        # Store the raw dataframe
        state['raw_dataframe'] = df
        
        # Store the first row separately as potential header
        if row_count > 0:
            header_row = df.iloc[0].tolist()
            state['header_row_if_present'] = header_row
            logger.debug(f"Potential header row: {header_row}")
            
            # Check for empty header values
            empty_headers = sum(1 for h in header_row if pd.isna(h) or str(h).strip() == '')
            if empty_headers > 0:
                logger.warning(f"Potential header row contains {empty_headers} empty values")
        else:
            logger.warning("CSV file is empty (0 rows)")
            state['header_row_if_present'] = []
            state['error_messages'].append("CSV file is empty")
    else:
        # If loading failed, set empty values
        logger.error(f"Failed to load CSV file: {csv_path}")
        state['raw_dataframe'] = None
        state['header_row_if_present'] = []
    
    return state

def detect_header_heuristic_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Apply heuristics to determine if the first row is likely a header.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with has_header_heuristic and potentially final_header
    """
    # Default to False
    state['has_header_heuristic'] = False
    
    # If raw_dataframe is None or empty, we can't detect headers
    if state['raw_dataframe'] is None or len(state['raw_dataframe']) < 2:
        state['error_messages'].append("Cannot detect headers: insufficient data")
        return state
    
    df = state['raw_dataframe']
    first_row = df.iloc[0]
    second_row = df.iloc[1]
    
    # Heuristic 1: Check if all values in the first row are strings
    all_strings_in_first_row = all(isinstance(val, str) for val in first_row)
    
    # Heuristic 2: Check if data types differ between first and second row
    type_differences = sum(type(first_row[i]) != type(second_row[i]) 
                          for i in range(len(first_row)) 
                          if pd.notna(first_row[i]) and pd.notna(second_row[i]))
    significant_type_diff = type_differences / len(first_row) > 0.5 if len(first_row) > 0 else False
    
    # Heuristic 3: Check for numeric values in first row (headers are rarely numeric)
    numeric_values_in_first_row = sum(isinstance(val, (int, float)) and not isinstance(val, bool) 
                                     for val in first_row if pd.notna(val))
    low_numeric_ratio = numeric_values_in_first_row / len(first_row) < 0.3 if len(first_row) > 0 else False
    
    # Heuristic 4: Check for common header keywords
    header_keywords = ['id', 'name', 'date', 'price', 'cost', 'email', 'phone', 'address', 'city', 'state', 'country']
    keyword_matches = sum(any(keyword in str(val).lower() for keyword in header_keywords) 
                         for val in first_row if pd.notna(val))
    has_header_keywords = keyword_matches / len(first_row) > 0.3 if len(first_row) > 0 else False
    
    # Combine heuristics
    state['has_header_heuristic'] = (all_strings_in_first_row and 
                                    (significant_type_diff or low_numeric_ratio or has_header_keywords))
    
    # If we detected a header, set it as the final header
    if state['has_header_heuristic']:
        # Convert all header values to strings
        state['final_header'] = [str(val) if pd.notna(val) else f"column_{i}" 
                                for i, val in enumerate(state['header_row_if_present'])]
    
    return state
