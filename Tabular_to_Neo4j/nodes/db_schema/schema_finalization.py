"""
Schema finalization module for database schema generation in the Tabular to Neo4j converter.
This module handles combining all intermediate results into the final Neo4j schema.
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def synthesize_final_schema_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Combine all intermediate results into the final Neo4j schema.
    Uses LLM to synthesize a comprehensive Neo4j schema from all intermediate analysis results.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with inferred_neo4j_schema
    """
    logger.info("Starting final schema synthesis process")
    
    # Validate required inputs
    missing_inputs = []
    if state.get('property_entity_mapping') is None:
        missing_inputs.append("property_entity_mapping")
    if state.get('entity_relationships') is None:
        missing_inputs.append("entity_relationships")
    if state.get('cypher_query_templates') is None:
        missing_inputs.append("cypher_query_templates")
    
    if missing_inputs:
        error_msg = f"Cannot synthesize final schema: missing required data: {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get all the intermediate results
        mapping = state['property_entity_mapping']
        relationships = state['entity_relationships']
        cypher_templates = state['cypher_query_templates']
        
        # Determine the main entity from the mapping (if any)
        entity_types = [item['entity_type'] for key, item in mapping.items() if item.get('type') == 'entity']
        main_entity = entity_types[0] if entity_types else 'Node'
        logger.info(f"Using {main_entity} as the main entity for schema finalization")
        
        # Extract entity types and their properties
        entities = {}
        for key, item in mapping.items():
            if item.get('type') == 'entity':
                entity_type = item.get('entity_type')
                entities[entity_type] = {
                    'is_primary': item.get('is_primary', False),
                    'properties': item.get('properties', [])
                }
        
        logger.debug(f"Synthesizing final schema for {len(entities)} entities and {len(relationships)} relationships")
        
        # Format the prompt with all schema information
        prompt = format_prompt('synthesize_final_schema.txt',
                              main_entity=main_entity,
                              entities=str(entities),
                              relationships=str(relationships),
                              cypher_templates=str(cypher_templates))
        
        # Call the LLM for final schema synthesis
        logger.debug("Calling LLM for final schema synthesis")
        try:
            response = call_llm_with_json_output(prompt, state_name="synthesize_final_schema")
            
            # Extract the final schema
            node_labels = response.get('node_labels', [])
            relationship_types = response.get('relationship_types', [])
            property_keys = response.get('property_keys', [])
            constraints = response.get('constraints', [])
            indexes = response.get('indexes', [])
            
            # Validate the schema elements
            if not node_labels:
                logger.warning("LLM did not generate any node labels, using entities from mapping")
                node_labels = [
                    {
                        'label': entity_type,
                        'description': f"Represents a {entity_type}",
                        'is_primary': entity_info.get('is_primary', False)
                    }
                    for entity_type, entity_info in entities.items()
                ]
            
            if not relationship_types:
                logger.warning("LLM did not generate any relationship types, using relationships from mapping")
                relationship_types = [
                    {
                        'type': rel.get('relationship_type', ''),
                        'source_label': rel.get('source_entity', ''),
                        'target_label': rel.get('target_entity', ''),
                        'description': f"Relates {rel.get('source_entity', '')} to {rel.get('target_entity', '')}",
                        'cardinality': rel.get('cardinality', 'MANY_TO_MANY')
                    }
                    for rel in relationships
                ]
            
            if not property_keys:
                logger.warning("LLM did not generate any property keys, extracting from entities")
                property_keys = []
                for entity_type, entity_info in entities.items():
                    for prop in entity_info.get('properties', []):
                        property_keys.append({
                            'key': prop.get('property_key', ''),
                            'data_type': 'STRING',  # Default to string
                            'description': f"Property of {entity_type}",
                            'belongs_to': entity_type,
                            'is_identifier': prop.get('is_identifier', False)
                        })
            
            # Create the final schema
            final_schema = {
                'node_labels': node_labels,
                'relationship_types': relationship_types,
                'property_keys': property_keys,
                'constraints': constraints,
                'indexes': indexes,
                'cypher_templates': cypher_templates
            }
            
            # Update the state
            state['inferred_neo4j_schema'] = final_schema
            
            logger.info("Generated final Neo4j schema successfully")
            
        except Exception as e:
            logger.error(f"Error synthesizing final schema: {str(e)}")
            
            # Create basic schema as fallback
            node_labels = [
                {
                    'label': entity_type,
                    'description': f"Represents a {entity_type}",
                    'is_primary': entity_info.get('is_primary', False)
                }
                for entity_type, entity_info in entities.items()
            ]
            
            relationship_types = [
                {
                    'type': rel.get('relationship_type', ''),
                    'source_label': rel.get('source_entity', ''),
                    'target_label': rel.get('target_entity', ''),
                    'description': f"Relates {rel.get('source_entity', '')} to {rel.get('target_entity', '')}",
                    'cardinality': rel.get('cardinality', 'MANY_TO_MANY')
                }
                for rel in relationships
            ]
            
            property_keys = []
            for entity_type, entity_info in entities.items():
                for prop in entity_info.get('properties', []):
                    property_keys.append({
                        'key': prop.get('property_key', ''),
                        'data_type': 'STRING',  # Default to string
                        'description': f"Property of {entity_type}",
                        'belongs_to': entity_type,
                        'is_identifier': prop.get('is_identifier', False)
                    })
            
            # Create constraints for identifier properties
            constraints = []
            for prop in property_keys:
                if prop.get('is_identifier', False):
                    entity_type = prop.get('belongs_to', '')
                    prop_key = prop.get('key', '')
                    if entity_type and prop_key:
                        constraints.append({
                            'type': 'UNIQUENESS',
                            'entity_type': entity_type,
                            'property_key': prop_key,
                            'description': f"Ensures {prop_key} is unique for {entity_type}"
                        })
            
            # Create the fallback schema
            fallback_schema = {
                'node_labels': node_labels,
                'relationship_types': relationship_types,
                'property_keys': property_keys,
                'constraints': constraints,
                'indexes': [],
                'cypher_templates': cypher_templates
            }
            
            # Update the state
            state['inferred_neo4j_schema'] = fallback_schema
            
            logger.warning("Using fallback schema due to LLM error")
            state['error_messages'].append(f"Error synthesizing final schema: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in final schema synthesis process: {str(e)}")
        state['error_messages'].append(f"Error in final schema synthesis process: {str(e)}")
        state['inferred_neo4j_schema'] = {
            'node_labels': [],
            'relationship_types': [],
            'property_keys': [],
            'constraints': [],
            'indexes': [],
            'cypher_templates': state.get('cypher_query_templates', {})
        }
    
    return state
