"""
Header application module for the Tabular to Neo4j converter.
This module handles applying the final headers to the DataFrame.
"""

from typing import Dict, Any, List
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def apply_header_node(state: GraphState, node_order: int) -> GraphState:
    """
    Apply the final headers to the DataFrame.
    
    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline
        
    Returns:
        Updated graph state with processed_dataframe
    """
    if state.get('raw_dataframe') is None or state.get('final_header') is None:
        error_msg = "Cannot apply header: missing raw dataframe or final header"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    logger.info("Applying final headers to the DataFrame")
    
    try:
        df = state['raw_dataframe']
        header = state['final_header']
        
        # Validate header length
        if len(header) != len(df.columns):
            error_msg = f"Header length ({len(header)}) does not match column count ({len(df.columns)})"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
            return state
        
        # Apply the header to the DataFrame
        processed_df = df.copy()
        processed_df.columns = header
        
        # If we detected a header in the original CSV, drop the first row
        if state.get('has_header_heuristic', False):
            logger.info("Dropping first row as it was detected as a header")
            processed_df = processed_df.iloc[1:].reset_index(drop=True)
        
        # Update the state with the processed DataFrame
        logger.info(f"Successfully applied headers to DataFrame with {len(processed_df)} rows and {len(processed_df.columns)} columns")
        state['processed_dataframe'] = processed_df
        
        # Log the first few rows with the new headers
        logger.debug(f"First few rows with new headers:\n{processed_df.head(3)}")
        
    except Exception as e:
        error_msg = f"Error applying headers: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    return state
