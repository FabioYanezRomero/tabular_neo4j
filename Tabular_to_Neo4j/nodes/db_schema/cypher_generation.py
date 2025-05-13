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
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        file_name = os.path.basename(state['csv_file_path'])
        
        # Extract entity types and their properties
        entities = {}
        for key, item in mapping.items():
            if item.get('type') == 'entity':
                entity_type = item.get('entity_type')
                entities[entity_type] = {
                    'is_primary': item.get('is_primary', False),
                    'properties': item.get('properties', [])
                }
        
        logger.debug(f"Preparing Cypher templates for {len(entities)} entities and {len(relationships)} relationships")
        
        # Format the prompt with schema information
        prompt = format_prompt('generate_cypher_templates.txt',
                              file_name=file_name,
                              primary_entity=primary_entity,
                              entities=str(entities),
                              relationships=str(relationships))
        
        # Call the LLM for Cypher template generation
        logger.debug("Calling LLM for Cypher template generation")
        try:
            response = call_llm_with_json_output(prompt, state_name="generate_cypher_templates")
            
            # Extract the Cypher templates
            load_query = response.get('load_query', '')
            constraint_queries = response.get('constraint_queries', [])
            index_queries = response.get('index_queries', [])
            example_queries = response.get('example_queries', [])
            
            # Validate the load query
            if not load_query:
                logger.warning("LLM did not generate a load query, creating a basic one")
                
                # Create a basic load query for the primary entity
                primary_props = []
                for entity_type, entity_info in entities.items():
                    if entity_info.get('is_primary', False):
                        primary_props = [
                            f"`{prop.get('column_name')}` AS `{prop.get('property_key')}`"
                            for prop in entity_info.get('properties', [])
                        ]
                
                load_query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{file_name}' AS row
                CREATE (:{primary_entity} {{
                    {', '.join(primary_props)}
                }})
                """
            
            # Update the state with the Cypher templates
            state['cypher_query_templates'] = {
                'load_query': load_query,
                'constraint_queries': constraint_queries,
                'index_queries': index_queries,
                'example_queries': example_queries
            }
            
            logger.info("Generated Cypher query templates successfully")
            
        except Exception as e:
            logger.error(f"Error generating Cypher templates: {str(e)}")
            
            # Create basic Cypher templates as fallback
            load_query_parts = []
            
            # Add CREATE statement for each entity
            for entity_type, entity_info in entities.items():
                if entity_info.get('is_primary', False):
                    props = [
                        f"`{prop.get('column_name')}` AS `{prop.get('property_key')}`"
                        for prop in entity_info.get('properties', [])
                    ]
                    
                    load_query_parts.append(f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{file_name}' AS row
                    CREATE (:{entity_type} {{
                        {', '.join(props)}
                    }})
                    """)
            
            # Add basic constraint for primary entity
            constraint_queries = []
            for entity_type, entity_info in entities.items():
                # Find an identifier property
                identifier_prop = next(
                    (prop.get('property_key') for prop in entity_info.get('properties', []) 
                     if prop.get('is_identifier', False)),
                    None
                )
                
                if identifier_prop:
                    constraint_queries.append(f"""
                    CREATE CONSTRAINT {entity_type}_{identifier_prop}_unique IF NOT EXISTS
                    FOR (n:{entity_type})
                    REQUIRE n.{identifier_prop} IS UNIQUE
                    """)
            
            # Update the state with the fallback templates
            state['cypher_query_templates'] = {
                'load_query': '\n'.join(load_query_parts),
                'constraint_queries': constraint_queries,
                'index_queries': [],
                'example_queries': []
            }
            
            logger.warning("Using fallback Cypher templates due to LLM error")
            state['error_messages'].append(f"Error generating Cypher templates: {str(e)}")
        
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
