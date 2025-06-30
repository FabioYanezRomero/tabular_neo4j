"""
CSV loader module for the Tabular to Neo4j converter.
This module handles loading CSV files into pandas DataFrames.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import load_csv_safely
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def load_csv_node(state: GraphState, node_order: int) -> GraphState:
    """
    Load a CSV file into a pandas DataFrame without assuming headers.
    
    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline
    Returns:
        Updated graph state with raw_dataframe and potential header
    """
    # Initialize error messages list if not present
    if 'error_messages' not in state:
        state['error_messages'] = []
    
    csv_path = state['csv_file_path']
    logger.info(f"Loading CSV file: {csv_path}")
    
    result_state = state
    try:
        # Load CSV file safely using utility function
        df, potential_header, encoding_used = load_csv_safely(csv_path)
        
        if df is None:
            error_msg = f"Failed to load CSV file: {csv_path}"
            logger.error(error_msg)
            result_state['error_messages'].append(error_msg)
        else:
            # Log successful loading
            logger.info(f"Successfully loaded CSV file")
            logger.debug(f"CSV file with {len(df)} rows and {len(df.columns)} columns using encoding: {encoding_used}")
            logger.debug(f"First few rows of data: {df.head(3)}")
            
            # Store the raw dataframe and potential header in the state
            result_state['raw_dataframe'] = df
            result_state['potential_header'] = potential_header
            result_state['encoding_used'] = encoding_used
            
            # If we have a potential header, also store it as the current header candidate
            if potential_header:
                logger.debug(f"Potential header detected: {potential_header}")
                result_state['current_header'] = potential_header
            
            # Log column data types
            dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
            logger.debug(f"Column data types: {dtypes_dict}")
            
            # Check for potential issues
            null_counts = df.isnull().sum()
            columns_with_nulls = [col for col, count in null_counts.items() if count > 0]
            if columns_with_nulls:
                logger.warning(f"Columns with null values: {columns_with_nulls}")
                result_state['columns_with_nulls'] = columns_with_nulls
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate rows in the CSV file")
                result_state['duplicate_row_count'] = duplicate_count
    except Exception as e:
        error_msg = f"Error loading CSV file: {str(e)}"
        logger.error(error_msg)
        result_state['error_messages'].append(error_msg)
    
    if not isinstance(result_state, GraphState):
        result_state = GraphState.from_dict(dict(result_state))
    return result_state
