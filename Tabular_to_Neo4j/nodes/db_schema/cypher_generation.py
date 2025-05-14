"""
Cypher generation module for database schema generation in the Tabular to Neo4j converter.
This module handles generating Cypher query templates.
"""

from typing import Dict, Any, List
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
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
            try:
                response = call_llm_with_json_output(prompt, state_name="generate_cypher_templates")
                logger.debug(f"LLM response for Cypher template generation: {response}")
            except Exception as e:
                logger.warning(f"Error calling LLM for Cypher template generation: {str(e)}")
                response = {}
            
            # Extract the Cypher templates from the response
            entity_creation_queries = []
            relationship_creation_queries = []
            example_queries = []
            
            # Log the raw response for debugging
            logger.debug(f"Raw response type: {type(response)}")
            if isinstance(response, dict):
                logger.debug(f"Response keys: {response.keys()}")
            elif isinstance(response, str):
                logger.debug(f"Response string: {response[:100]}..." if len(response) > 100 else response)
            
            # Handle different response formats
            if isinstance(response, dict):
                # Try to extract entity_creation_queries
                if 'entity_creation_queries' in response:
                    entity_queries = response['entity_creation_queries']
                    if isinstance(entity_queries, list):
                        entity_creation_queries = entity_queries
                    elif isinstance(entity_queries, str):
                        # Try to convert string to list if it's a string representation
                        try:
                            import json
                            # Clean the string if needed
                            if entity_queries.strip().startswith('[') and entity_queries.strip().endswith(']'):
                                entity_creation_queries = json.loads(entity_queries)
                            else:
                                # Try to find a list in the string
                                import re
                                list_match = re.search(r'\[(.*?)\]', entity_queries)
                                if list_match:
                                    try:
                                        entity_creation_queries = json.loads(f"[{list_match.group(1)}]")
                                    except:
                                        logger.warning("Could not parse extracted list from entity_creation_queries")
                        except Exception as e:
                            logger.warning(f"Could not parse entity_creation_queries as JSON: {str(e)}")
                            
                # If we still don't have entity_creation_queries, try to find them in the response
                if not entity_creation_queries:
                    # Check if any key contains 'entity' and 'queries'
                    for key in response.keys():
                        if 'entity' in key.lower() and ('queries' in key.lower() or 'creation' in key.lower()):
                            try:
                                value = response[key]
                                if isinstance(value, list):
                                    entity_creation_queries = value
                                    logger.info(f"Found entity creation queries under key '{key}'")
                                    break
                            except:
                                continue
                
                # Try to extract relationship_creation_queries
                if 'relationship_creation_queries' in response:
                    rel_queries = response['relationship_creation_queries']
                    if isinstance(rel_queries, list):
                        relationship_creation_queries = rel_queries
                    elif isinstance(rel_queries, str):
                        try:
                            import json
                            # Clean the string if needed
                            if rel_queries.strip().startswith('[') and rel_queries.strip().endswith(']'):
                                relationship_creation_queries = json.loads(rel_queries)
                            else:
                                # Try to find a list in the string
                                import re
                                list_match = re.search(r'\[(.*?)\]', rel_queries)
                                if list_match:
                                    try:
                                        relationship_creation_queries = json.loads(f"[{list_match.group(1)}]")
                                    except:
                                        logger.warning("Could not parse extracted list from relationship_creation_queries")
                        except Exception as e:
                            logger.warning(f"Could not parse relationship_creation_queries as JSON: {str(e)}")
                            
                # If we still don't have relationship_creation_queries, try to find them in the response
                if not relationship_creation_queries:
                    # Check if any key contains 'relationship' and 'queries'
                    for key in response.keys():
                        if 'relationship' in key.lower() and ('queries' in key.lower() or 'creation' in key.lower()):
                            try:
                                value = response[key]
                                if isinstance(value, list):
                                    relationship_creation_queries = value
                                    logger.info(f"Found relationship creation queries under key '{key}'")
                                    break
                            except:
                                continue
                
                # Try to extract example_queries
                if 'example_queries' in response:
                    ex_queries = response['example_queries']
                    if isinstance(ex_queries, list):
                        example_queries = ex_queries
                    elif isinstance(ex_queries, str):
                        try:
                            import json
                            example_queries = json.loads(ex_queries)
                        except:
                            logger.warning("Could not parse example_queries as JSON")
            
            # Log the results
            logger.info(f"Generated {len(entity_creation_queries)} entity creation queries, {len(relationship_creation_queries)} relationship queries, and {len(example_queries)} example queries")
            
            # Validate that we have at least some templates
            if not entity_creation_queries and not relationship_creation_queries:
                logger.warning("LLM did not generate any Cypher templates, using fallback mechanism")
                
                # Get entity types from the property-entity mapping
                property_entity_mapping = state.get('property_entity_mapping', {})
                entities_with_properties = {}
                
                # Extract entities and their properties
                for entity_type, entity_info in property_entity_mapping.items():
                    if entity_info.get('type') == 'entity':
                        entities_with_properties[entity_type] = entity_info.get('properties', [])
                
                # Get entity types and their properties from the property-entity mapping
                property_entity_mapping = state.get('property_entity_mapping', {})
                if not property_entity_mapping:
                    # If property_entity_mapping is not available, try to get it from reconciled_classification
                    reconciled_classification = state.get('reconciled_classification', {})
                    entities = {}
                    properties = {}
                    
                    # Separate entities and properties
                    for column_name, info in reconciled_classification.items():
                        if info.get('classification') == 'entity':
                            entities[column_name] = {
                                'type': 'entity',
                                'entity_type': column_name,
                                'properties': []
                            }
                        elif info.get('classification') == 'property':
                            # Assign to first entity as fallback
                            if entities:
                                first_entity = next(iter(entities.keys()))
                                if 'properties' not in entities[first_entity]:
                                    entities[first_entity]['properties'] = []
                                entities[first_entity]['properties'].append(column_name)
                    
                    property_entity_mapping = entities
                    logger.info(f"Created fallback property-entity mapping with {len(entities)} entities")
                
                # Generate fallback entity creation queries
                for entity_type, entity_info in property_entity_mapping.items():
                    if entity_info.get('type') == 'entity':
                        # Get properties for this entity
                        properties = entity_info.get('properties', [])
                        
                        # Create a basic Cypher query for this entity
                        if properties:
                            property_str = ", ".join([f"{prop}: ${prop}" for prop in properties])
                            query = f"CREATE (e:{entity_type} {{id: $id, {property_str}}})"
                        else:
                            query = f"CREATE (e:{entity_type} {{id: $id}})"
                            
                        entity_creation_queries.append({"query": query})
                        logger.info(f"Generated fallback entity creation query for {entity_type} with {len(properties)} properties")
                
                # Generate fallback relationship queries if we have multiple entities
                if len(property_entity_mapping) > 1:
                    entity_types = [entity_type for entity_type, info in property_entity_mapping.items() 
                                  if info.get('type') == 'entity']
                    
                    # Create relationships between consecutive entities
                    for i in range(len(entity_types) - 1):
                        source = entity_types[i]
                        target = entity_types[i + 1]
                        query = f"MATCH (a:{source} {{id: $id_a}}), (b:{target} {{id: $id_b}}) CREATE (a)-[:RELATED_TO]->(b)"
                        relationship_creation_queries.append({"query": query})
                        logger.info(f"Generated fallback relationship query between {source} and {target}")
                
                # Generate basic example queries
                example_queries.append({"query": "MATCH (n) RETURN n LIMIT 10"})
                example_queries.append({"query": "MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 10"})
                logger.info("Generated fallback example queries")
                
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
