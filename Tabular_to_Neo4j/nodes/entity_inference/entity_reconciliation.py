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
    if state.get('entity_property_classification') is None:
        error_msg = "Cannot reconcile entity/property classifications: missing classification data"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get the initial classification
        classification = state['entity_property_classification']
        
        # Initialize consensus dictionary
        consensus = {}
        
        # Process each column using LLM for reconciliation
        for column_name, info in classification.items():
            # Get analytics and semantics for this column
            analytics = state.get('column_analytics', {}).get(column_name, {})
            semantics = state.get('llm_column_semantics', {}).get(column_name, {})
            
            # Skip if we don't have both analytics and semantics
            if not analytics or not semantics:
                consensus[column_name] = info  # Use initial classification as fallback
                continue
            
            # Get sample values for this column
            sample_values = []
            if state.get('processed_dataframe') is not None:
                sample_values = state['processed_dataframe'][column_name].dropna().sample(
                    min(5, len(state['processed_dataframe'][column_name].dropna()))
                ).tolist() if len(state['processed_dataframe'][column_name].dropna()) > 0 else []
            
            # Determine analytics-based classification
            analytics_classification = "entity_property"  # Default
            uniqueness = analytics.get('uniqueness', 0)
            cardinality = analytics.get('cardinality', 0)
            
            if uniqueness > UNIQUENESS_THRESHOLD:
                analytics_classification = "entity_identifier"
                logger.debug(f"Analytics suggests '{column_name}' is an entity_identifier (uniqueness: {uniqueness:.2f})")

            elif analytics.get('cardinality', 0) < len(state['processed_dataframe']) * 0.1 and analytics.get('cardinality', 0) > 1:
                analytics_classification = "new_entity_type"
            
            # Format the prompt with column information
            prompt = format_prompt('reconcile_entity_property.txt',
                                  column_name=column_name,
                                  initial_classification=str(info),
                                  analytics_classification=analytics_classification,
                                  llm_classification=semantics.get('neo4j_role', 'UNKNOWN'),
                                  sample_values=str(sample_values))
            
            # Call the LLM for reconciliation
            try:
                response = call_llm_with_json_output(prompt, state_name="reconcile_entity_property")
                
                # Extract the reconciliation results
                consensus_classification = response.get('consensus_classification', 'property')  # Default to property
                entity_type = response.get('entity_type', '')
                property_of = response.get('property_of', '')
                neo4j_property_key = response.get('neo4j_property_key', to_neo4j_property_name(column_name))
                confidence = response.get('confidence', 0.5)
                reasoning = response.get('reasoning', 'No reasoning provided')
                
                # Create consensus entry
                consensus[column_name] = {
                    'column_name': column_name,
                    'classification': consensus_classification,
                    'entity_type': entity_type if consensus_classification == 'entity' else '',
                    'property_of': property_of if consensus_classification == 'property' else '',
                    'neo4j_property_key': neo4j_property_key if consensus_classification == 'property' else '',
                    'semantic_type': info.get('semantic_type', 'Unknown'),
                    'confidence': confidence,
                    'reasoning': reasoning
                }
                
                # Add uniqueness information for entities to help with later processing
                if consensus_classification == 'entity':
                    consensus[column_name]['uniqueness_ratio'] = analytics.get('uniqueness_ratio', 0)
                
            except Exception as e:
                logger.error(f"Error reconciling column {column_name}: {str(e)}")
                consensus[column_name] = info  # Use initial classification as fallback
        
        # Second pass: for properties without a clear entity association, try to infer which entity they belong to
        for column_name, info in consensus.items():
            if info['classification'] == 'property' and (not info.get('property_of') or info.get('property_of') == ''):
                # Try to find the associated entity
                associated_entity = None
                
                # Look for entities with similar naming patterns
                column_parts = column_name.split('_')
                for entity_col, entity_info in consensus.items():
                    if entity_info['classification'] == 'entity':
                        entity_name = entity_info['entity_type'].lower()
                        # Check if any part of the column name matches the entity name
                        if any(part.lower() == entity_name for part in column_parts):
                            associated_entity = entity_info
                            break
                
                if associated_entity:
                    info['property_of'] = associated_entity['entity_type']
                    logger.info(f"Associated property '{column_name}' with entity '{associated_entity['entity_type']}' based on naming pattern")
                else:
                    # If we can't find an association, leave property_of empty
                    info['note'] = "No clear entity association found"
        
        # Update the state
        state['entity_property_consensus'] = consensus
        
    except Exception as e:
        logger.error(f"Error reconciling entity/property classifications: {str(e)}")
        state['error_messages'].append(f"Error reconciling entity/property classifications: {str(e)}")
        state['entity_property_consensus'] = state.get('entity_property_classification', {})
    
    return state
