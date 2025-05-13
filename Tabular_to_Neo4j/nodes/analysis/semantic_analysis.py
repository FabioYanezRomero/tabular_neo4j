"""
Semantic analysis module for the Tabular to Neo4j converter.
This module handles semantic analysis of columns using LLM.
"""

from typing import Dict, Any, List
import pandas as pd
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

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
        missing = []
        if state.get('processed_dataframe') is None:
            missing.append("processed_dataframe")
        if state.get('column_analytics') is None:
            missing.append("column_analytics")
            
        error_msg = f"Cannot perform semantic analysis: missing required data: {', '.join(missing)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    logger.info("Using LLM to analyze semantic meaning of columns")
    
    try:
        df = state['processed_dataframe']
        analytics = state['column_analytics']
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Initialize semantic analysis results
        semantic_results = {}
        
        # Process each column
        for column_name in df.columns:
            logger.debug(f"Analyzing semantic meaning of column '{column_name}'")
            
            # Get column analytics
            column_analytics = analytics.get(column_name, {})
            
            # Get sample values for this column
            sample_values = df[column_name].dropna().sample(
                min(5, len(df[column_name].dropna()))
            ).tolist() if len(df[column_name].dropna()) > 0 else []
            
            # Get file name for the prompt
            file_name = os.path.basename(state.get('csv_file_path', 'unknown.csv'))
            
            # Get metadata for the CSV file
            metadata = get_metadata_for_state(state)
            metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
            
            # Format the prompt with column information and metadata
            prompt = format_prompt('analyze_column_semantics.txt',
                                  file_name=file_name,
                                  column_name=column_name,
                                  data_type=column_analytics.get('data_type', 'unknown'),
                                  sample_values=str(sample_values),
                                  uniqueness=column_analytics.get('uniqueness', 0),
                                  cardinality=column_analytics.get('cardinality', 0),
                                  primary_entity=primary_entity,
                                  patterns=column_analytics.get('patterns', []),
                                  metadata_text=metadata_text)
            
            # Call the LLM for semantic analysis
            try:
                response = call_llm_with_json_output(prompt, state_name="analyze_column_semantics")
                
                # Extract the semantic analysis results
                semantic_type = response.get('semantic_type', 'Unknown')
                neo4j_role = response.get('neo4j_role', 'UNKNOWN')
                description = response.get('description', '')
                related_entity = response.get('related_entity', '')
                
                # Store the results
                semantic_results[column_name] = {
                    'semantic_type': semantic_type,
                    'neo4j_role': neo4j_role,
                    'description': description,
                    'related_entity': related_entity
                }
                
                logger.info(f"Column '{column_name}' semantic type: {semantic_type}, Neo4j role: {neo4j_role}")
                
            except Exception as e:
                logger.error(f"Error in LLM semantic analysis for column '{column_name}': {str(e)}")
                
                # Use a fallback classification
                semantic_results[column_name] = {
                    'semantic_type': 'Unknown',
                    'neo4j_role': 'PROPERTY',
                    'description': f'Error in LLM analysis: {str(e)}',
                    'related_entity': ''
                }
                
                state['error_messages'].append(f"Error in semantic analysis for column '{column_name}': {str(e)}")
        
        # Update the state with the semantic analysis results
        state['llm_column_semantics'] = semantic_results
        
    except Exception as e:
        error_msg = f"Error in semantic column analysis: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    return state
