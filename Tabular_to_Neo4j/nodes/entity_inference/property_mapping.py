"""
Property mapping module for entity inference in the Tabular to Neo4j converter.
This module handles mapping properties to their respective entities.
"""

from typing import Dict, Any, List
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def map_properties_to_entities_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Map properties to their respective entities and create a clear property-entity mapping.
    Uses LLM to determine the best entity for each property.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with property_entity_mapping
    """
    logger.info("Starting property-to-entity mapping process")
    
    # Validate required inputs
    if state.get('entity_property_consensus') is None:
        error_msg = "Cannot map properties to entities: missing consensus data"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get the consensus classification
        consensus = state['entity_property_consensus']
        
        # Initialize property-entity mapping
        property_entity_mapping = {}
        
        # First pass: identify all entity types and their properties
        for column_name, info in consensus.items():
            classification = info.get('classification', '')
            
            if classification == 'entity':
                # This column represents an entity
                entity_type = info.get('entity_type', '')
                if entity_type and entity_type not in property_entity_mapping:
                    # Initialize the entity in our mapping
                    property_entity_mapping[entity_type] = []
                    logger.debug(f"Identified entity type: {entity_type}")
            
            elif classification == 'property':
                # This column represents a property
                property_of = info.get('property_of', '')
                if property_of:
                    # If we know which entity this property belongs to
                    if property_of not in property_entity_mapping:
                        property_entity_mapping[property_of] = []
                    
                    # Add the property to its entity
                    property_entity_mapping[property_of].append({
                        'column_name': column_name,
                        'property_key': info.get('neo4j_property_key', to_neo4j_property_name(column_name))
                    })
                    logger.debug(f"Mapped property '{column_name}' to entity '{property_of}'")
                else:
                    logger.warning(f"Property '{column_name}' has no associated entity")

        
        logger.debug(f"Identified {len(property_entity_mapping)} entity types: {', '.join(property_entity_mapping.keys())}")
        
        # Second pass: handle any properties that don't have a clear entity association
        # Try to infer entity association based on column name patterns
        for column_name, info in consensus.items():
            if info.get('classification') == 'property' and not info.get('property_of'):
                # Try to find an entity type that matches part of the column name
                column_parts = column_name.lower().split('_')
                
                for entity_type in property_entity_mapping.keys():
                    # Check if any part of the column name matches the entity type
                    if entity_type.lower() in column_parts:
                        # Add the property to this entity
                        property_entity_mapping[entity_type].append({
                            'column_name': column_name,
                            'property_key': info.get('neo4j_property_key', to_neo4j_property_name(column_name))
                        })
                        logger.info(f"Inferred that property '{column_name}' belongs to entity '{entity_type}' based on naming pattern")
                        break
        
        # For each entity type, use LLM to validate and refine property mapping
        final_property_entity_mapping = {}
        
        for entity_type, properties in property_entity_mapping.items():
            # Skip if this is not a list (might be a dict from a previous iteration)
            if not isinstance(properties, list):
                final_property_entity_mapping[entity_type] = properties
                continue
                
            if not properties:
                logger.warning(f"No properties found for entity type '{entity_type}', skipping")
                continue
            
            logger.info(f"Mapping {len(properties)} properties to entity type '{entity_type}'")
            
            # Format the prompt with entity and property information
            prompt = format_prompt('map_properties_to_entity.txt',
                                  entity_type=entity_type,
                                  properties=str(properties))
            
            # Call the LLM for property mapping
            try:
                response = call_llm_with_json_output(prompt, state_name="map_properties_to_entity")
                
                # Extract the mapping results
                mapped_properties = response.get('properties', [])
                
                if not mapped_properties:
                    logger.warning(f"LLM returned no property mappings for entity '{entity_type}', using original properties")
                    mapped_properties = [
                        {
                            'column_name': prop['column_name'],
                            'property_key': prop['property_key']
                        }
                        for prop in properties
                    ]
                
                # Create entity mapping entry
                final_property_entity_mapping[entity_type] = {
                    'type': 'entity',
                    'entity_type': entity_type,
                    'properties': mapped_properties
                }
                
            except Exception as e:
                logger.error(f"Error mapping properties for entity '{entity_type}': {str(e)}")
                
                # Use original properties as fallback
                final_property_entity_mapping[entity_type] = {
                    'type': 'entity',
                    'entity_type': entity_type,
                    'properties': [
                        {
                            'column_name': prop['column_name'],
                            'property_key': prop['property_key']
                        }
                        for prop in properties
                    ]
                }
                
                state['error_messages'].append(f"Error mapping properties for entity '{entity_type}': {str(e)}")
        
        # Update the state
        state['property_entity_mapping'] = final_property_entity_mapping
        
    except Exception as e:
        logger.error(f"Error mapping properties to entities: {str(e)}")
        state['error_messages'].append(f"Error mapping properties to entities: {str(e)}")
        state['property_entity_mapping'] = {}
    
    return state
