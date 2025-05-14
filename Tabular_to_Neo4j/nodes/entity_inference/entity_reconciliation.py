"""
Entity reconciliation module for entity inference in the Tabular to Neo4j converter.
This module handles reconciling different classification approaches.
"""

from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def reconcile_entity_property_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Reconcile analytics-based and LLM-based classifications
    to create a consensus model of entities and properties.
    Uses LLM to reconcile different classification approaches.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with entity_property_consensus
    """
    logger.info("Starting entity-property reconciliation process")
    
    # Validate required inputs
    missing_inputs = []
    if state.get('entity_property_classification') is None:
        missing_inputs.append("entity_property_classification")
    if state.get('rule_based_classification') is None:
        missing_inputs.append("rule_based_classification")
        
    if missing_inputs:
        error_msg = f"Cannot reconcile entity/property classifications: missing required data: {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get the classifications
        llm_classification = state['entity_property_classification']
        rule_based_classification = state['rule_based_classification']
        
        # Initialize consensus dictionary
        consensus = {}
        
        # Process each column to determine if reconciliation is needed
        for column_name in llm_classification.keys():
            # Get both classifications for this column
            llm_info = llm_classification.get(column_name, {})
            rule_info = rule_based_classification.get(column_name, {})
            
            # Skip if we don't have both classifications
            if not llm_info or not rule_info:
                logger.warning(f"Missing classification for column '{column_name}', using LLM classification as fallback")
                consensus[column_name] = llm_info if llm_info else {'column_name': column_name, 'classification': 'property', 'confidence': 0.5}
                continue
            
            # Check if there's a discrepancy between the classifications
            llm_classification_result = llm_info.get('classification', '')
            rule_classification_result = rule_info.get('classification', '')
            
            # If classifications match, and are not empty no need for reconciliation
            if llm_classification_result == rule_classification_result:
                logger.info(f"Classifications match for column '{column_name}': {llm_classification_result}")                
                consensus[column_name] = llm_info
                continue
            
            # If there's a discrepancy, perform reconciliation
            logger.info(f"Classification discrepancy for '{column_name}': LLM={llm_classification_result}, Rule={rule_classification_result}")
            
            # Get sample values for this column
            sample_values = []
            if state.get('processed_dataframe') is not None:
                sample_values = state['processed_dataframe'][column_name].dropna().sample(
                    min(5, len(state['processed_dataframe'][column_name].dropna()))
                ).tolist() if len(state['processed_dataframe'][column_name].dropna()) > 0 else []
            
            # Get metadata for the CSV file
            metadata = get_metadata_for_state(state)
            metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
            
            # Format the prompt with column information for reconciliation
            prompt = format_prompt('reconcile_entity_property.txt',
                                  column_name=column_name,
                                  metadata_text=metadata_text,
                                  analytics_classification=rule_classification_result,
                                  llm_classification=llm_classification_result,
                                  sample_values=str(sample_values))
            
            # Call the LLM for reconciliation
            try:
                response = call_llm_with_json_output(prompt, state_name="reconcile_entity_property")
                
                # Extract the reconciliation results
                consensus_classification = response.get('consensus_classification', 'property')  # Default to property
                confidence = response.get('confidence', 0.5)
                
                # Create consensus entry
                consensus[column_name] = {
                    'column_name': column_name,
                    'classification': consensus_classification,
                    'confidence': confidence
                }
                
                logger.info(f"Reconciled classification for '{column_name}': {consensus_classification} with confidence {confidence}")
                
                # Add uniqueness information for entities to help with later processing
                if consensus_classification == 'entity':
                    consensus[column_name]['uniqueness_ratio'] = analytics.get('uniqueness_ratio', 0)
                
            except Exception as e:
                logger.error(f"Error reconciling column {column_name}: {str(e)}")
                consensus[column_name] = info  # Use initial classification as fallback
        
        # Add uniqueness information for entities to help with later processing
        for column_name, info in consensus.items():
            if info['classification'] == 'entity':
                analytics = state.get('column_analytics', {}).get(column_name, {})
                info['uniqueness_ratio'] = analytics.get('uniqueness_ratio', 0)
        
        # Update the state
        state['entity_property_consensus'] = consensus
        
    except Exception as e:
        logger.error(f"Error reconciling entity/property classifications: {str(e)}")
        state['error_messages'].append(f"Error reconciling entity/property classifications: {str(e)}")
        state['entity_property_consensus'] = state.get('entity_property_classification', {})
    
    return state
