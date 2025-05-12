"""
Input nodes for the LangGraph CSV analysis pipeline.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import load_csv_safely
import logging

logger = logging.getLogger(__name__)


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
    
    # Load the CSV file
    df, errors = load_csv_safely(state['csv_file_path'], header=None)
    
    # Add any errors to the state
    if errors:
        state['error_messages'].extend(errors)
        
    if df is not None:
        # Store the raw dataframe
        state['raw_dataframe'] = df
        
        # Store the first row separately as potential header
        if len(df) > 0:
            state['header_row_if_present'] = df.iloc[0].tolist()
        else:
            state['header_row_if_present'] = []
            state['error_messages'].append("CSV file is empty")
    else:
        # If loading failed, set empty values
        logger.error("Failed to load CSV file: %s", state['csv_file_path'])
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
