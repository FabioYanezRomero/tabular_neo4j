"""
Column analysis nodes for the LangGraph CSV analysis pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.analytics_utils import analyze_all_columns
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output, call_llm_with_state
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS

def perform_column_analytics_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Perform statistical and pattern analysis on each column.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with column_analytics
    """
    if state.get('processed_dataframe') is None:
        state['error_messages'].append("Cannot analyze columns: no processed dataframe available")
        return state
    
    try:
        # Analyze all columns in the processed dataframe
        analytics_results = analyze_all_columns(state['processed_dataframe'])
        
        # Update the state
        state['column_analytics'] = analytics_results
        
    except Exception as e:
        state['error_messages'].append(f"Error analyzing columns: {str(e)}")
        state['column_analytics'] = {}
    
    return state

def llm_semantic_column_analysis_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to analyze the semantic meaning of each column and its role in Neo4j modeling.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with llm_column_semantics
    """
    if state.get('processed_dataframe') is None or state.get('column_analytics') is None:
        state['error_messages'].append("Cannot perform semantic analysis: missing processed dataframe or column analytics")
        return state
    
    # Initialize the semantic analysis results
    state['llm_column_semantics'] = {}
    
    # Get the table subject (primary entity) from the filename
    table_subject = get_primary_entity_from_filename(state['csv_file_path'])
    
    # Process each column
    for column_name in state['processed_dataframe'].columns:
        try:
            # Get the column analytics
            column_analytics = state['column_analytics'].get(column_name, {})
            
            # Get sample values
            sample_values = state['processed_dataframe'][column_name].dropna().sample(
                min(5, len(state['processed_dataframe'][column_name].dropna()))
            ).tolist() if len(state['processed_dataframe'][column_name].dropna()) > 0 else []
            
            # Format sample values for the prompt
            sample_values_str = str(sample_values)
            
            # Format the patterns for the prompt
            patterns_str = str(column_analytics.get('patterns', {}))
            
            # Format the prompt with the column information
            prompt = format_prompt('analyze_column_semantic.txt',
                                  column_name=column_name,
                                  table_subject=table_subject,
                                  sample_values=sample_values_str,
                                  uniqueness_ratio=column_analytics.get('uniqueness_ratio', 0),
                                  cardinality=column_analytics.get('cardinality', 0),
                                  data_type=column_analytics.get('data_type', 'unknown'),
                                  missing_percentage=column_analytics.get('missing_percentage', 0) * 100,
                                  patterns=patterns_str)
            
            # Call the LLM for the semantic_analysis state and parse the JSON response
            response = call_llm_with_json_output(prompt, state_name="semantic_analysis")
            
            # Store the semantic analysis result
            state['llm_column_semantics'][column_name] = {
                'column_name': column_name,
                'semantic_type': response.get('semantic_type', 'Unknown'),
                'neo4j_role': response.get('neo4j_role', 'UNKNOWN'),
                'new_node_label_suggestion': response.get('new_node_label_suggestion', ''),
                'relationship_type_suggestion': response.get('relationship_type_suggestion', ''),
                'reasoning': response.get('reasoning', '')
            }
            
        except Exception as e:
            state['error_messages'].append(f"Error analyzing column {column_name}: {str(e)}")
            # Add a default entry for failed columns
            state['llm_column_semantics'][column_name] = {
                'column_name': column_name,
                'semantic_type': 'Unknown',
                'neo4j_role': 'UNKNOWN',
                'new_node_label_suggestion': '',
                'relationship_type_suggestion': '',
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    return state
