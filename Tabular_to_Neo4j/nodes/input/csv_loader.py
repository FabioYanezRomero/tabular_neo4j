"""
CSV loader module for the Tabular to Neo4j converter.
This module handles loading CSV files into pandas DataFrames.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import load_csv_safely
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
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
    
    try:
        # Load CSV file safely using utility function
        df, potential_header, encoding_used = load_csv_safely(csv_path)
        
        if df is None:
            error_msg = f"Failed to load CSV file: {csv_path}"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
            return state
        
        # Log successful loading
        logger.info(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns using encoding: {encoding_used}")
        logger.debug(f"First few rows of data: {df.head(3)}")
        
        # Store the raw dataframe and potential header in the state
        state['raw_dataframe'] = df
        state['potential_header'] = potential_header
        state['encoding_used'] = encoding_used
        
        # If we have a potential header, also store it as the current header candidate
        if potential_header:
            logger.debug(f"Potential header detected: {potential_header}")
            state['current_header'] = potential_header
        
        # Log column data types
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        logger.debug(f"Column data types: {dtypes_dict}")
        
        # Check for potential issues
        null_counts = df.isnull().sum()
        columns_with_nulls = [col for col, count in null_counts.items() if count > 0]
        if columns_with_nulls:
            logger.warning(f"Columns with null values: {columns_with_nulls}")
            state['columns_with_nulls'] = columns_with_nulls
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows in the CSV file")
            state['duplicate_row_count'] = duplicate_count
        
    except Exception as e:
        error_msg = f"Error loading CSV file: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    return state
