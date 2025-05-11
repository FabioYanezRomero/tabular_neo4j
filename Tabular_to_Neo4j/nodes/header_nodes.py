"""
Header processing nodes for the LangGraph CSV analysis pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_utils import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import df_to_string_sample
from Tabular_to_Neo4j.config import TARGET_HEADER_LANGUAGE, MAX_SAMPLE_ROWS

def infer_header_llm_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to infer appropriate headers when no header is detected.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with inferred_header and final_header
    """
    if state['raw_dataframe'] is None:
        state['error_messages'].append("Cannot infer headers: no data available")
        return state
    
    # Get a sample of the data for the LLM
    df_sample = state['raw_dataframe'].head(MAX_SAMPLE_ROWS)
    data_sample = df_to_string_sample(df_sample)
    
    try:
        # Format the prompt with the data sample
        prompt = format_prompt('infer_header.txt', data_sample=data_sample)
        
        # Call the LLM and parse the JSON response
        response = call_llm_with_json_output(prompt)
        
        # Check if the response is a list
        if isinstance(response, list):
            inferred_header = response
        else:
            # Try to extract a list from the response dictionary
            for key in response:
                if isinstance(response[key], list):
                    inferred_header = response[key]
                    break
            else:
                raise ValueError("LLM response does not contain a list of headers")
        
        # Update the state
        state['inferred_header'] = inferred_header
        state['final_header'] = inferred_header
        
    except Exception as e:
        state['error_messages'].append(f"Error inferring headers: {str(e)}")
        # Create fallback headers
        state['inferred_header'] = [f"column_{i}" for i in range(len(state['raw_dataframe'].columns))]
        state['final_header'] = state['inferred_header']
    
    return state

def validate_header_llm_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to validate and potentially improve headers.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with validated_header and potentially updated final_header
    """
    if state['raw_dataframe'] is None or 'final_header' not in state or not state['final_header']:
        state['error_messages'].append("Cannot validate headers: missing data or headers")
        return state
    
    # Create a temporary DataFrame with the headers applied
    temp_df = state['raw_dataframe'].copy()
    
    # If the first row was detected as a header, remove it
    if state.get('has_header_heuristic', False):
        temp_df = temp_df.iloc[1:]
    
    # Apply the current headers
    temp_df.columns = state['final_header']
    
    # Get a sample of the data for the LLM
    df_sample = temp_df.head(MAX_SAMPLE_ROWS)
    data_sample = df_to_string_sample(df_sample)
    
    try:
        # Format the prompt with the headers and data sample
        prompt = format_prompt('validate_header.txt', 
                              headers=state['final_header'],
                              data_sample=data_sample)
        
        # Call the LLM and parse the JSON response
        response = call_llm_with_json_output(prompt)
        
        # Extract the validation results
        is_correct = response.get('is_correct', True)
        validated_header = response.get('validated_header', state['final_header'])
        suggestions = response.get('suggestions', "")
        
        # Update the state
        state['is_header_correct_llm'] = is_correct
        state['validated_header'] = validated_header
        state['header_correction_suggestions'] = suggestions
        
        # If the LLM suggested changes, update the final header
        if not is_correct:
            state['final_header'] = validated_header
        
    except Exception as e:
        state['error_messages'].append(f"Error validating headers: {str(e)}")
        # Keep the existing headers
        state['is_header_correct_llm'] = True
        state['validated_header'] = state['final_header']
        state['header_correction_suggestions'] = ""
    
    return state

def translate_header_llm_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to translate headers to the target language if needed.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with translated_header and potentially updated final_header
    """
    if 'final_header' not in state or not state['final_header']:
        state['error_messages'].append("Cannot translate headers: no headers available")
        return state
    
    try:
        # Format the prompt with the headers and target language
        prompt = format_prompt('translate_header.txt', 
                              headers=state['final_header'],
                              target_language=TARGET_HEADER_LANGUAGE)
        
        # Call the LLM and parse the JSON response
        response = call_llm_with_json_output(prompt, is_translation=True)
        
        # Extract the translation results
        is_in_target_language = response.get('is_in_target_language', True)
        translated_header = response.get('translated_header', state['final_header'])
        
        # Update the state
        state['is_header_in_target_language'] = is_in_target_language
        state['translated_header'] = translated_header
        
        # If translation was needed, update the final header
        if not is_in_target_language:
            state['final_header'] = translated_header
        
    except Exception as e:
        state['error_messages'].append(f"Error translating headers: {str(e)}")
        # Keep the existing headers
        state['is_header_in_target_language'] = True
        state['translated_header'] = state['final_header']
    
    return state

def apply_header_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Apply the final headers to the DataFrame.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with processed_dataframe
    """
    if state['raw_dataframe'] is None or 'final_header' not in state or not state['final_header']:
        state['error_messages'].append("Cannot apply headers: missing data or headers")
        return state
    
    try:
        # Create a copy of the raw dataframe
        processed_df = state['raw_dataframe'].copy()
        
        # If the first row was detected as a header, remove it
        if state.get('has_header_heuristic', False):
            processed_df = processed_df.iloc[1:]
        
        # Apply the final headers
        processed_df.columns = state['final_header']
        
        # Update the state
        state['processed_dataframe'] = processed_df
        
    except Exception as e:
        state['error_messages'].append(f"Error applying headers: {str(e)}")
        # Create a fallback processed dataframe
        if state['raw_dataframe'] is not None:
            fallback_df = state['raw_dataframe'].copy()
            fallback_headers = [f"column_{i}" for i in range(len(fallback_df.columns))]
            fallback_df.columns = fallback_headers
            state['processed_dataframe'] = fallback_df
        else:
            state['processed_dataframe'] = None
    
    return state
