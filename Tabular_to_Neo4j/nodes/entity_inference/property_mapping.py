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
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Initialize property-entity mapping
        property_entity_mapping = {}
        
        # Extract entity types and their properties
        entity_types = set([primary_entity])
        entity_properties: Dict[str, List[Dict[str, Any]]] = {primary_entity: []}
        
        # First pass: identify all entity types and their direct properties
        for column_name, info in consensus.items():
            classification = info.get('classification', '')
            entity_type = info.get('entity_type', primary_entity)
            
            # Add to entity types set
            if classification in ['entity_identifier', 'new_entity_type']:
                entity_types.add(entity_type)
                if entity_type not in entity_properties:
                    entity_properties[entity_type] = []
            
            # Map properties to their entity types
            if classification in ['entity_property', 'entity_identifier']:
                if entity_type not in entity_properties:
                    entity_properties[entity_type] = []
                
                entity_properties[entity_type].append({
                    'column_name': column_name,
                    'property_key': info.get('neo4j_property_key', to_neo4j_property_name(column_name)),
                    'classification': classification
                })
        
        logger.debug(f"Identified {len(entity_types)} entity types: {', '.join(entity_types)}")
        
        # Second pass: handle secondary entity properties and relationship properties
        for column_name, info in consensus.items():
            classification = info.get('classification', '')
            
            if classification == 'secondary_entity_property':
                entity_type = info.get('entity_type', primary_entity)
                
                if entity_type not in entity_properties:
                    entity_properties[entity_type] = []
                
                entity_properties[entity_type].append({
                    'column_name': column_name,
                    'property_key': info.get('neo4j_property_key', to_neo4j_property_name(column_name)),
                    'classification': 'entity_property'  # Treat as regular entity property for mapping
                })
            
            elif classification == 'relationship_property':
                # Get relationship info
                relationship_type = info.get('relationship_type', '')
                source_entity = info.get('source_entity', '')
                target_entity = info.get('target_entity', '')
                
                if not relationship_type or not source_entity or not target_entity:
                    logger.warning(f"Incomplete relationship info for column '{column_name}', skipping")
                    continue
                
                # Create a key for this relationship
                rel_key = f"{source_entity}__{relationship_type}__{target_entity}"
                
                if rel_key not in property_entity_mapping:
                    property_entity_mapping[rel_key] = {
                        'type': 'relationship',
                        'relationship_type': relationship_type,
                        'source_entity': source_entity,
                        'target_entity': target_entity,
                        'properties': []
                    }
                
                property_entity_mapping[rel_key]['properties'].append({
                    'column_name': column_name,
                    'property_key': info.get('neo4j_property_key', to_neo4j_property_name(column_name))
                })
        
        # For each entity type, use LLM to validate and refine property mapping
        for entity_type, properties in entity_properties.items():
            if not properties:
                logger.warning(f"No properties found for entity type '{entity_type}', skipping")
                continue
            
            logger.info(f"Mapping {len(properties)} properties to entity type '{entity_type}'")
            
            # Format the prompt with entity and property information
            prompt = format_prompt('map_properties_to_entity.txt',
                                  entity_type=entity_type,
                                  properties=str(properties),
                                  primary_entity=primary_entity,
                                  is_primary=(entity_type == primary_entity))
            
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
                            'property_key': prop['property_key'],
                            'is_identifier': prop['classification'] == 'entity_identifier'
                        }
                        for prop in properties
                    ]
                
                # Create entity mapping entry
                property_entity_mapping[entity_type] = {
                    'type': 'entity',
                    'entity_type': entity_type,
                    'is_primary': entity_type == primary_entity,
                    'properties': mapped_properties
                }
                
                # Check if we have an identifier property
                has_identifier = any(prop.get('is_identifier', False) for prop in mapped_properties)
                if not has_identifier:
                    logger.warning(f"No identifier property found for entity '{entity_type}'")
                    
                    # Try to find a suitable identifier
                    for prop in mapped_properties:
                        if 'id' in prop.get('property_key', '').lower() or 'key' in prop.get('property_key', '').lower():
                            logger.info(f"Setting '{prop.get('property_key')}' as identifier for entity '{entity_type}' based on name")
                            prop['is_identifier'] = True
                            break
                
            except Exception as e:
                logger.error(f"Error mapping properties for entity '{entity_type}': {str(e)}")
                
                # Use a fallback mapping
                property_entity_mapping[entity_type] = {
                    'type': 'entity',
                    'entity_type': entity_type,
                    'is_primary': entity_type == primary_entity,
                    'properties': [
                        {
                            'column_name': prop['column_name'],
                            'property_key': prop['property_key'],
                            'is_identifier': prop['classification'] == 'entity_identifier'
                        }
                        for prop in properties
                    ]
                }
                
                state['error_messages'].append(f"Error mapping properties for entity '{entity_type}': {str(e)}")
        
        # Update the state
        state['property_entity_mapping'] = property_entity_mapping
        
    except Exception as e:
        logger.error(f"Error mapping properties to entities: {str(e)}")
        state['error_messages'].append(f"Error mapping properties to entities: {str(e)}")
        state['property_entity_mapping'] = {}
    
    return state
