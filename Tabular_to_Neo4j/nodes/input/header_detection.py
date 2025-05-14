"""
Header detection module for the Tabular to Neo4j converter.
This module handles detecting headers in CSV files using heuristics.
"""

from typing import Dict, Any, List
import pandas as pd
import re
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def detect_header_heuristic_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Apply heuristics to determine if the first row is likely a header.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with has_header_heuristic and potentially final_header
    """
    if state.get('raw_dataframe') is None:
        error_msg = "Cannot detect header: no raw dataframe available"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    logger.info("Applying heuristics to detect if the first row is a header")
    
    try:
        df = state['raw_dataframe']
        potential_header = state.get('potential_header', [])
        
        # Initialize with default assumption
        has_header = False
        
        # If we have a potential header from the CSV loader, analyze it
        if potential_header and len(potential_header) == len(df.columns):
            logger.debug(f"Analyzing potential header: {potential_header}")
            
            # Check if the potential header has different data types than the rest of the data
            first_row_types = [type(x) for x in df.iloc[0].tolist()]
            rest_data_types = [df[col].iloc[1:].dtype for col in df.columns]
            
            type_mismatch_count = sum(1 for i, t1 in enumerate(first_row_types) 
                                     if str(t1) != str(rest_data_types[i]))
            
            # Check if the potential header has string patterns typical of headers
            header_pattern_count = sum(1 for h in potential_header 
                                      if isinstance(h, str) and 
                                      (re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', h) or 
                                       re.match(r'^[A-Za-z_][A-Za-z0-9_\s]*$', h)))
            
            # Make decision based on heuristics
            if (type_mismatch_count / len(df.columns) > 0.5 or 
                header_pattern_count / len(potential_header) > 0.7):
                has_header = True
                logger.info(f"Heuristics suggest the first row is a header (type mismatches: {type_mismatch_count}, header patterns: {header_pattern_count})")
            else:
                logger.info(f"Heuristics suggest the first row is NOT a header (type mismatches: {type_mismatch_count}, header patterns: {header_pattern_count})")
        
        # Update the state with our findings
        state['has_header_heuristic'] = has_header
        
        # If we determined there is a header, set it as the final header
        if has_header and potential_header:
            logger.info(f"Setting final header based on heuristic detection: {potential_header}")
            state['final_header'] = potential_header
        
    except Exception as e:
        error_msg = f"Error detecting header: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        state['has_header_heuristic'] = False
    
    return state
