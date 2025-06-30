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

def detect_header_heuristic_node(state: GraphState, node_order: int) -> GraphState:
    """
    Apply heuristics to determine if the first row is likely a header.
    
    Args:
        state: The current graph state    
        node_order: The order of the node in the pipeline
    Returns:
        Updated graph state with has_header_heuristic and potentially final_header
    """
    if state.get('raw_dataframe') is None:
        error_msg = "Cannot detect header: no raw dataframe available"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        # Continue to end for unified return

    
    logger.info("Applying heuristics to detect if the first row is a header")
    
    try:
        df = state['raw_dataframe']
        
        # Get the first row as potential header if not already in state
        potential_header = state.get('potential_header', [])
        if not potential_header and len(df) > 0:
            potential_header = df.iloc[0].tolist()
            logger.debug(f"Using first row as potential header: {potential_header}")
        
        # Initialize with default assumption
        has_header = False
        
        # If we have a potential header, analyze it
        if potential_header and len(potential_header) == len(df.columns):
            logger.debug(f"Analyzing potential header: {potential_header}")
            
            # Check if the potential header has different data types than the rest of the data
            if len(df) > 1:  # Need at least 2 rows to compare
                first_row_types = [type(x) for x in df.iloc[0].tolist()]
                rest_data_types = [df[col].iloc[1:].dtype for col in df.columns]
                
                type_mismatch_count = sum(1 for i, t1 in enumerate(first_row_types) 
                                        if str(t1) != str(rest_data_types[i]))
                type_mismatch_ratio = type_mismatch_count / len(df.columns) if len(df.columns) > 0 else 0
                logger.debug(f"Type mismatch ratio: {type_mismatch_ratio}")
            else:
                type_mismatch_ratio = 0
            
            # Check if the potential header has string patterns typical of headers
            header_pattern_count = sum(1 for h in potential_header 
                                      if isinstance(h, str) and 
                                      (re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', h) or 
                                       re.match(r'^[A-Za-z_][A-Za-z0-9_\s]*$', h) or
                                       re.match(r'^[A-Za-z_][A-Za-z0-9_\-]*$', h)))
            header_pattern_ratio = header_pattern_count / len(potential_header) if len(potential_header) > 0 else 0
            logger.debug(f"Header pattern ratio: {header_pattern_ratio}")
            
            # Additional check: Are column names snake_case or camelCase?
            snake_case_count = sum(1 for h in potential_header 
                                  if isinstance(h, str) and '_' in h)
            camel_case_count = sum(1 for h in potential_header 
                                  if isinstance(h, str) and not '_' in h and 
                                  re.match(r'^[a-z]+[A-Z]', h))
            case_pattern_ratio = (snake_case_count + camel_case_count) / len(potential_header) if len(potential_header) > 0 else 0
            logger.debug(f"Case pattern ratio: {case_pattern_ratio}")
            
            # Make decision based on heuristics
            # Lower the threshold for header patterns to catch more headers
            if (type_mismatch_ratio > 0.3 or 
                header_pattern_ratio > 0.5 or
                case_pattern_ratio > 0.3):
                has_header = True
                logger.info(f"Heuristics suggest the first row is a header (type mismatches: {type_mismatch_ratio:.2f}, header patterns: {header_pattern_ratio:.2f}, case patterns: {case_pattern_ratio:.2f})")
            else:
                logger.info(f"Heuristics suggest the first row is NOT a header (type mismatches: {type_mismatch_ratio:.2f}, header patterns: {header_pattern_ratio:.2f}, case patterns: {case_pattern_ratio:.2f})")
                
            # Special case: If all potential header items are strings and look like column names,
            # override the decision
            all_strings = all(isinstance(h, str) for h in potential_header)
            all_look_like_headers = all(re.match(r'^[A-Za-z_][A-Za-z0-9_\s\-]*$', h) if isinstance(h, str) else False for h in potential_header)
            if all_strings and all_look_like_headers:
                has_header = True
                logger.info("Overriding decision: All potential header items are strings and look like column names")
                
            # Special case: Check for common header names
            common_header_terms = ['id', 'name', 'date', 'email', 'price', 'cost', 'amount', 'quantity', 'address', 'phone']
            common_term_count = sum(1 for h in potential_header 
                                  if isinstance(h, str) and any(term in h.lower() for term in common_header_terms))
            common_term_ratio = common_term_count / len(potential_header) if len(potential_header) > 0 else 0
            if common_term_ratio > 0.3:
                has_header = True
                logger.info(f"Overriding decision: {common_term_ratio:.2f} of potential header items contain common header terms")
        
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
    
    if not isinstance(state, GraphState):
        state = GraphState.from_dict(dict(state))
    return state
