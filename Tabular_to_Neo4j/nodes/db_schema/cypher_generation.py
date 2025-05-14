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
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS
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
        
        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
        
        # Get sample data from the processed dataframe
        sample_data = ""
        if state.get('processed_dataframe') is not None:
            df = state['processed_dataframe']
            if not df.empty:
                try:
                    sample_data = df.head(MAX_SAMPLE_ROWS).to_dict(orient='records')
                    sample_data = str(sample_data)  # Convert to string for the prompt
                except Exception as e:
                    logger.warning(f"Error converting dataframe to dict: {str(e)}")
                    # Fallback to string representation
                    sample_data = df.head(MAX_SAMPLE_ROWS).to_string(index=False)
        
        # Format the prompt with schema information
        prompt = format_prompt('generate_cypher_templates.txt',
                              entities=str(entities),
                              relationships=str(relationships),
                              metadata_text=metadata_text,
                              sample_data=sample_data)
        
        # Call the LLM for Cypher template generation
        logger.info("Calling LLM for Cypher template generation with unique identifiers")
        try:
            response = call_llm_with_json_output(prompt, state_name="generate_cypher_templates")
            
            # Extract the Cypher templates from the new response format
            entity_creation_queries = response.get('entity_creation_queries', [])
            relationship_creation_queries = response.get('relationship_creation_queries', [])
            example_queries = response.get('example_queries', [])
            
            # Log the results
            logger.info(f"Generated {len(entity_creation_queries)} entity creation queries, {len(relationship_creation_queries)} relationship queries, and {len(example_queries)} example queries")
            
            # Validate that we have at least some templates
            if not entity_creation_queries and not relationship_creation_queries:
                logger.warning("LLM did not generate any Cypher templates")
                
            # Ensure each entity has a unique identifier in the creation queries
            for entity_type, entity_info in entities.items():
                # Check if any template creates this entity with a UUID
                has_uuid_creation = any(
                    entity_type in q.get('query', '') and 'UUID' in q.get('query', '')
                    for q in entity_creation_queries
                )
                
                if not has_uuid_creation and entity_creation_queries:
                    logger.warning(f"No UUID generation found for entity {entity_type} in templates")
                    # We'll let the LLM handle this, but log a warning

            
            # Update the state with the Cypher templates using the new format
            state['cypher_query_templates'] = {
                'entity_creation_queries': entity_creation_queries,
                'relationship_queries': relationship_creation_queries,
                'example_queries': example_queries,
                'constraint_queries': []  # Keep this for backward compatibility
            }
            
            # Add default UUID constraints for each entity if not already included
            for entity_type in entities.keys():
                constraint_query = f"CREATE CONSTRAINT ON (e:{entity_type}) ASSERT e.uuid IS UNIQUE"
                state['cypher_query_templates']['constraint_queries'].append({
                    'entity_type': entity_type,
                    'property': 'uuid',
                    'query': constraint_query
                })
            
            logger.info("Generated Cypher query templates successfully")
            
        except Exception as e:
            logger.error(f"Error generating Cypher templates: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in Cypher template generation process: {str(e)}")
        state['error_messages'].append(f"Error in Cypher template generation process: {str(e)}")
        state['cypher_query_templates'] = {
            'entity_creation_queries': [],
            'relationship_queries': [],
            'example_queries': [],
            'constraint_queries': []
        }
    
    return state
