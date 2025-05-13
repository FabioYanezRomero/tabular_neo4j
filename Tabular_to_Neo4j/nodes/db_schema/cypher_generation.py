"""
Cypher generation module for database schema generation in the Tabular to Neo4j converter.
This module handles generating Cypher query templates.
"""

from typing import Dict, Any, List
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def generate_cypher_templates_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Generate template Cypher queries for the inferred schema.
    Uses LLM to create Cypher query templates for loading and querying the Neo4j graph model.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with cypher_query_templates
    """
    logger.info("Starting Cypher template generation process")
    
    # Validate required inputs
    missing_inputs = []
    if state.get('property_entity_mapping') is None:
        missing_inputs.append("property_entity_mapping")
    if state.get('entity_relationships') is None:
        missing_inputs.append("entity_relationships")
    
    if missing_inputs:
        error_msg = f"Cannot generate Cypher templates: missing required data: {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get the property-entity mapping and relationships
        mapping = state['property_entity_mapping']
        relationships = state['entity_relationships']
        
        # Extract entity types and their properties
        entities = {}
        for key, item in mapping.items():
            if item.get('type') == 'entity':
                entity_type = item.get('entity_type')
                
                # Always add a unique identifier property that doesn't depend on existing properties
                properties = item.get('properties', [])
                
                # Every entity needs a generated UUID as a unique identifier
                entities[entity_type] = {
                    'is_primary': item.get('is_primary', False),
                    'properties': properties,
                    'needs_generated_id': True  # Always generate a unique ID for every entity
                }
        
        logger.debug(f"Preparing Cypher templates for {len(entities)} entities and {len(relationships)} relationships")
        
        # Format the prompt with schema information
        prompt = format_prompt('generate_cypher_templates.txt',
                              entities=str(entities),
                              relationships=str(relationships))
        
        # Call the LLM for Cypher template generation
        logger.info("Calling LLM for Cypher template generation with unique identifiers")
        try:
            response = call_llm_with_json_output(prompt, state_name="generate_cypher_templates")
            
            # Extract the Cypher templates
            cypher_templates = response.get('cypher_templates', [])
            constraints_and_indexes = response.get('constraints_and_indexes', [])
            
            # Log the results
            logger.info(f"Generated {len(cypher_templates)} Cypher templates and {len(constraints_and_indexes)} constraints/indexes")
            
            # Validate that we have at least some templates
            if not cypher_templates:
                logger.warning("LLM did not generate any Cypher templates")
                
            # Ensure each entity has a unique identifier
            for entity_type, entity_info in entities.items():
                # Check if any constraint is created for this entity's UUID
                has_uuid_constraint = any(
                    c.get('entity_type') == entity_type and 'uuid' in c.get('property', '').lower()
                    for c in constraints_and_indexes
                )
                
                if not has_uuid_constraint:
                    logger.info(f"Adding UUID constraint for entity {entity_type}")
                    constraints_and_indexes.append({
                        "type": "CONSTRAINT",
                        "entity_type": entity_type,
                        "property": "uuid",
                        "query": f"CREATE CONSTRAINT ON (e:{entity_type}) ASSERT e.uuid IS UNIQUE"
                    })
                    
                # Check if any template creates this entity with a UUID
                has_uuid_creation = any(
                    entity_type in t.get('query', '') and 'randomUUID()' in t.get('query', '')
                    for t in cypher_templates
                )
                
                if not has_uuid_creation and cypher_templates:
                    logger.warning(f"No UUID generation found for entity {entity_type} in templates")
                    # We'll let the LLM handle this, but log a warning

            
            # Update the state with the Cypher templates
            state['cypher_query_templates'] = {
                'entity_creation_queries': [],
                'relationship_queries': [],
                'constraint_queries': [],
                'example_queries': []
            }
            
            # Process the cypher templates based on their purpose
            for template in cypher_templates:
                purpose = template.get('purpose', '').lower()
                query = template.get('query', '')
                description = template.get('description', '')
                
                if not query:
                    continue
                    
                if 'entity' in purpose and 'creat' in purpose:
                    state['cypher_query_templates']['entity_creation_queries'].append({
                        'query': query,
                        'description': description
                    })
                elif 'relation' in purpose:
                    state['cypher_query_templates']['relationship_queries'].append({
                        'query': query,
                        'description': description
                    })
                elif 'query' in purpose or 'find' in purpose or 'match' in purpose:
                    state['cypher_query_templates']['example_queries'].append({
                        'query': query,
                        'description': description
                    })
            
            # Process constraints and indexes
            for constraint in constraints_and_indexes:
                if constraint.get('type', '').upper() == 'CONSTRAINT':
                    state['cypher_query_templates']['constraint_queries'].append({
                        'entity_type': constraint.get('entity_type', ''),
                        'property': constraint.get('property', ''),
                        'query': constraint.get('query', '')
                    })
            
            logger.info("Generated Cypher query templates successfully")
            
        except Exception as e:
            logger.error(f"Error generating Cypher templates: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in Cypher template generation process: {str(e)}")
        state['error_messages'].append(f"Error in Cypher template generation process: {str(e)}")
        state['cypher_query_templates'] = {
            'load_query': '',
            'constraint_queries': [],
            'index_queries': [],
            'example_queries': []
        }
    
    return state
