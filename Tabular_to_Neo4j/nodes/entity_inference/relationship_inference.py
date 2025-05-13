"""
Relationship inference module for entity inference in the Tabular to Neo4j converter.
This module handles inferring relationships between entity types.
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def infer_entity_relationships_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Infer relationships between entities based on property mapping and entity types.
    Uses LLM to determine the most likely relationships between entities.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with entity_relationships
    """
    logger.info("Starting entity relationship inference process")
    
    # Validate required inputs
    if state.get('property_entity_mapping') is None:
        error_msg = "Cannot infer entity relationships: missing property-entity mapping"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get the property-entity mapping
        mapping = state['property_entity_mapping']
        
        # Get column analytics if available
        column_analytics = state.get('column_analytics', {})
        
        # Extract entity types with their properties and analytics
        entities_with_properties = []
        for entity_type, entity_data in mapping.items():
            if entity_data.get('type') == 'entity':
                properties = entity_data.get('properties', [])
                
                # Enhance properties with analytics data if available
                enhanced_properties = []
                for prop in properties:
                    column_name = prop.get('column_name', '')
                    enhanced_prop = prop.copy()
                    
                    # Add analytics data if available
                    if column_name in column_analytics:
                        analytics = column_analytics[column_name]
                        enhanced_prop['analytics'] = {
                            'data_type': analytics.get('data_type', 'unknown'),
                            'uniqueness_ratio': analytics.get('uniqueness_ratio', 0),
                            'cardinality': analytics.get('cardinality', 0),
                            'missing_percentage': analytics.get('missing_percentage', 0),
                            'sample_values': analytics.get('sample_values', [])
                        }
                    
                    enhanced_properties.append(enhanced_prop)
                
                entities_with_properties.append({
                    'entity_type': entity_type,
                    'properties': enhanced_properties
                })
        
        entity_types = [entity['entity_type'] for entity in entities_with_properties]
        logger.info(f"Found {len(entity_types)} entity types for relationship inference: {', '.join(entity_types)}")
        
        # If we have 0 or 1 entity types, there are no relationships to infer
        if len(entity_types) <= 1:
            logger.info(f"Found {len(entity_types)} entity types, no relationships to infer")
            state['entity_relationships'] = []
            return state
        
        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
        
        # Get the processed dataframe for sample data if available
        sample_data = ""
        if state.get('processed_dataframe') is not None:
            df = state['processed_dataframe']
            if not df.empty:
                # Get a small sample of the data (first 5 rows)
                sample_data = df.head(5).to_string(index=False)
        
        # Format the prompt with entity information and metadata
        prompt = format_prompt('infer_entity_relationships.txt',
                              entities=str(entities_with_properties),
                              metadata_text=metadata_text,
                              sample_data=sample_data)
        
        # Call the LLM for relationship inference
        logger.info("Calling LLM to infer relationships between entities")
        response = call_llm_with_json_output(prompt, state_name="infer_entity_relationships")
        
        # Extract the inferred relationships
        inferred_relationships = response.get('relationships', [])
        reasoning = response.get('reasoning', 'No reasoning provided')
        
        logger.info(f"LLM inferred {len(inferred_relationships)} relationships between entities")
        logger.debug(f"LLM reasoning: {reasoning}")
        
        # Update the state with the inferred relationships
        state['entity_relationships'] = inferred_relationships
        
    except Exception as e:
        logger.error(f"Error inferring entity relationships: {str(e)}")
        state['error_messages'].append(f"Error inferring entity relationships: {str(e)}")
        state['entity_relationships'] = []
    
    return state
