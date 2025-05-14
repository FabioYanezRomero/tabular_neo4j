"""
Header validation module for the Tabular to Neo4j converter.
This module handles validating and improving headers.
"""

from typing import Dict, Any, List
import pandas as pd
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import df_to_json_sample
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def validate_header_llm_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to validate and potentially improve headers.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with validated_header and potentially updated final_header
    """
    if state.get('raw_dataframe') is None or state.get('final_header') is None:
        error_msg = "Cannot validate header: missing raw dataframe or header"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    logger.info("Using LLM to validate and improve headers")
    
    try:
        df = state['raw_dataframe']
        current_header = state['final_header']
        
        # Get a sample of the data for the LLM
        sample_rows = min(MAX_SAMPLE_ROWS, len(df))
        data_sample = df_to_json_sample(df, sample_rows)
        
        # Get file name for the prompt
        file_name = os.path.basename(state.get('csv_file_path', 'unknown.csv'))
        
        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
        
        # Format the prompt with the data sample, current header, and metadata
        prompt = format_prompt('validate_header.txt',
                              file_name=file_name,
                              data_sample=data_sample,
                              current_header=current_header,
                              column_count=len(df.columns),
                              row_count=len(df),
                              metadata_text=metadata_text)
        
        # Call the LLM to validate headers
        logger.debug("Calling LLM for header validation")
        response = call_llm_with_json_output(prompt, state_name="validate_header")
        
        # Extract the validation results
        is_correct = response.get('is_correct', False)
        validated_header = response.get('validated_header', current_header)
        suggestions = response.get('suggestions', '')
        
        # Validate the response
        if not isinstance(validated_header, list):
            error_msg = f"LLM did not return a list of headers: {validated_header}"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
            return state
        
        if len(validated_header) != len(df.columns):
            error_msg = f"LLM returned {len(validated_header)} headers, but CSV has {len(df.columns)} columns"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
            return state
        
        # Update the state with the validated headers
        state['validated_header'] = validated_header
        
        # If the LLM suggested improvements, update the final header
        if not is_correct:
            logger.info(f"LLM suggested header improvements: {suggestions}")
            logger.info(f"Updated headers: {validated_header}")
            state['final_header'] = validated_header
        else:
            logger.info("LLM confirmed headers are appropriate")
        
    except Exception as e:
        error_msg = f"Error validating headers: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    return state
