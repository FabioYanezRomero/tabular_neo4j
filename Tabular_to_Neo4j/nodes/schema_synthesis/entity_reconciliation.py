"""
Entity reconciliation module for schema synthesis in the Tabular to Neo4j converter.
This module handles the second step in schema synthesis: reconciling different classification approaches.
"""

from typing import Dict, Any, Set
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.nodes.schema_synthesis.utils import (
    to_neo4j_property_name,
    find_associated_entity_type,
    find_associated_relationship
)
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def reconcile_entity_property_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Second step in schema synthesis: Reconcile analytics-based and LLM-based classifications
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
        file_name = os.path.basename(state['csv_file_path'])
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Initialize consensus dictionary
        consensus = {}
        
        # Track entity types for later use
        entity_types = set([primary_entity])
        
        # First pass: identify all entity types
        for column_name, info in classification.items():
            if info['classification'] in ['new_entity_type']:
                entity_types.add(info['entity_type'])
        
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
                                  file_name=file_name,
                                  column_name=column_name,
                                  primary_entity=primary_entity,
                                  initial_classification=str(info),
                                  analytics_classification=analytics_classification,
                                  llm_classification=semantics.get('neo4j_role', 'UNKNOWN'),
                                  sample_values=str(sample_values),
                                  entity_types=str(list(entity_types)))
            
            # Call the LLM for reconciliation
            try:
                response = call_llm_with_json_output(prompt, state_name="reconcile_entity_property")
                
                # Extract the reconciliation results
                consensus_classification = response.get('consensus_classification', info['classification'])
                entity_type = response.get('entity_type', info.get('entity_type', primary_entity))
                relationship_to_primary = response.get('relationship_to_primary', info.get('relationship_to_primary', ''))
                neo4j_property_key = response.get('neo4j_property_key', info.get('neo4j_property_key', to_neo4j_property_name(column_name)))
                confidence = response.get('confidence', 0.5)
                reasoning = response.get('reasoning', '')
                
                # Create consensus entry
                consensus[column_name] = {
                    'column_name': column_name,
                    'classification': consensus_classification,
                    'entity_type': entity_type,
                    'relationship_to_primary': relationship_to_primary,
                    'neo4j_property_key': neo4j_property_key,
                    'semantic_type': info.get('semantic_type', 'Unknown'),
                    'confidence': confidence,
                    'reasoning': reasoning
                }
                
                # Add additional fields based on classification type
                if consensus_classification == 'entity_identifier':
                    consensus[column_name]['uniqueness_ratio'] = analytics.get('uniqueness_ratio', 0)
                
                elif consensus_classification == 'new_entity_type':
                    # Ensure we have a relationship type
                    if not relationship_to_primary:
                        consensus[column_name]['relationship_to_primary'] = f"HAS_{entity_type.upper()}"
                    
                    consensus[column_name]['primary_entity'] = primary_entity
                
                # Add entity type to our set if it's a new one
                if consensus_classification == 'new_entity_type':
                    entity_types.add(entity_type)
                
            except Exception as e:
                logger.error(f"Error reconciling column {column_name}: {str(e)}")
                consensus[column_name] = info  # Use initial classification as fallback
        
        # Second pass: resolve secondary entity properties
        for column_name, info in consensus.items():
            if info['classification'] == 'secondary_entity_property':
                # Try to find the associated entity type
                associated_entity = find_associated_entity_type(
                    column_name, 
                    info,
                    consensus,
                    entity_types,
                    state['llm_column_semantics']
                )
                
                if associated_entity:
                    info['entity_type'] = associated_entity
                else:
                    # If we can't find an association, default to primary entity
                    info['entity_type'] = primary_entity
                    info['note'] = "No clear entity association found, defaulted to primary entity"
        
        # Third pass: resolve relationship properties
        for column_name, info in consensus.items():
            if info['classification'] == 'relationship_property':
                # Try to find the associated relationship
                relationship_info = find_associated_relationship(
                    column_name,
                    info,
                    consensus
                )
                
                if relationship_info:
                    info['relationship_type'] = relationship_info['relationship']
                    info['source_entity'] = relationship_info['source_entity']
                    info['target_entity'] = relationship_info['target_entity']
                else:
                    # If we can't find an association, convert to primary entity property
                    info['classification'] = 'entity_property'
                    info['entity_type'] = primary_entity
                    info['note'] = "Originally classified as relationship property but no association found"
        
        # Update the state
        state['entity_property_consensus'] = consensus
        
    except Exception as e:
        logger.error(f"Error reconciling entity/property classifications: {str(e)}")
        state['error_messages'].append(f"Error reconciling entity/property classifications: {str(e)}")
        state['entity_property_consensus'] = state.get('entity_property_classification', {})
    
    return state
