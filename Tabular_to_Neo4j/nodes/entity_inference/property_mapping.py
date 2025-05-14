"""
Property mapping module for entity inference in the Tabular to Neo4j converter.
This module handles mapping properties to their respective entities.
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows
from Tabular_to_Neo4j.utils.analytics_utils import analyze_column
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS

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
        # Get the consensus classification and other necessary data
        consensus = state['entity_property_consensus']
        column_analytics = state.get('column_analytics', {})
        processed_dataframe = state.get('processed_dataframe')
        metadata = get_metadata_for_state(state)
        
        # Extract entities and properties from consensus
        entities = []
        properties = []
        
        for column_name, info in consensus.items():
            classification = info.get('classification', '')
            
            if classification == 'entity':
                entity_type = info.get('entity_type', '')
                if entity_type:
                    entities.append({
                        'entity_type': entity_type,
                        'column_name': column_name
                    })
                    logger.debug(f"Identified entity type: {entity_type} from column '{column_name}'")
            
            elif classification == 'property':
                # Get column analytics or analyze the column if not available
                column_data = column_analytics.get(column_name, {})
                if not column_data and processed_dataframe is not None and column_name in processed_dataframe.columns:
                    column_data = analyze_column(processed_dataframe[column_name])
                
                properties.append({
                    'column_name': column_name,
                    'property_key': info.get('neo4j_property_key', to_neo4j_property_name(column_name)),
                    'sample_values': column_data.get('sample_values', []),
                    'analytics': column_data
                })
                logger.debug(f"Identified property: {column_name}")
        
        logger.info(f"Found {len(entities)} entities and {len(properties)} properties")
        
        # If we have no properties, nothing to map
        if not properties:
            logger.warning("No properties found, nothing to map")
            state['property_entity_mapping'] = {entity['entity_type']: {'type': 'entity', 'entity_type': entity['entity_type'], 'properties': []} for entity in entities}
            return state
            
        # If we have no entities, raise an error
        if not entities:
            error_msg = "No entities detected but properties exist. Cannot map properties without entities."
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
            state['property_entity_mapping'] = {}
            return state
            
        # If we have only one entity, all properties belong to it
        if len(entities) == 1:
            entity_type = entities[0]['entity_type']
            logger.info(f"Only one entity detected ({entity_type}), assigning all properties to it")
            
            # Create the property-entity mapping with all properties assigned to the single entity
            property_entity_mapping = {
                entity_type: {
                    'type': 'entity',
                    'entity_type': entity_type,
                    'properties': [
                        {
                            'column_name': prop['column_name'],
                            'property_key': prop['property_key']
                        } for prop in properties
                    ]
                }
            }
            
            # Update the state and return
            state['property_entity_mapping'] = property_entity_mapping
            return state
            
        # Multiple entities detected, initialize property-entity mapping with LLMs
        logger.info(f"Multiple entities detected ({len(entities)}), will map each property to an entity")
        
        # Initialize the property-entity mapping with empty properties lists
        final_property_entity_mapping = {}
        for entity in entities:
            entity_type = entity['entity_type']
            final_property_entity_mapping[entity_type] = {
                'type': 'entity',
                'entity_type': entity_type,
                'properties': []
            }
        
        # Prepare metadata for the prompt
        metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
        
        # Process each property and determine which entity it belongs to
        for prop in properties:
            column_name = prop['column_name']
            property_key = prop['property_key']
            logger.info(f"Processing property: {column_name}")
            
            # Get sample values for this column
            sample_values = []
            if processed_dataframe is not None and column_name in processed_dataframe.columns:
                sample_values = processed_dataframe[column_name].dropna().sample(
                    min(MAX_SAMPLE_ROWS, len(processed_dataframe[column_name].dropna()))
                ).tolist() if len(processed_dataframe[column_name].dropna()) > 0 else []
            
            # Format the prompt for this specific property
            prompt = format_prompt('map_properties_to_entity.txt',
                                  property=str(prop),
                                  entities=str(entities),
                                  metadata_text=metadata_text,
                                  sample_values=str(sample_values),
                                  analytics=str(prop.get('analytics', {})))
            
            # Call the LLM to determine which entity this property belongs to
            logger.info(f"Calling LLM to determine which entity property '{column_name}' belongs to")
            
            try:
                response = call_llm_with_json_output(prompt, state_name=f"map_property_{column_name}")
                
                # Extract the entity this property belongs to
                entity_type = response.get('entity', '')
                confidence = response.get('confidence', 0.5)
                
                # Validate that the entity exists in our mapping
                if entity_type and entity_type in final_property_entity_mapping:
                    # Add property to the appropriate entity
                    final_property_entity_mapping[entity_type]['properties'].append({
                        'column_name': column_name,
                        'property_key': property_key,
                        'confidence': confidence
                    })
                    logger.info(f"Mapped property '{column_name}' to entity '{entity_type}' with confidence {confidence}")
                else:
                    # If entity not found, assign to the first entity
                    if entities:
                        first_entity = entities[0]['entity_type']
                        final_property_entity_mapping[first_entity]['properties'].append({
                            'column_name': column_name,
                            'property_key': property_key,
                            'confidence': 0.3  # Lower confidence since this is a fallback
                        })
                        logger.warning(f"LLM returned unknown entity '{entity_type}' for property '{column_name}', assigned to '{first_entity}' with low confidence")
                    else:
                        logger.error(f"Cannot map property '{column_name}': no entities available and LLM returned unknown entity '{entity_type}'")
                
            except Exception as e:
                logger.error(f"Error mapping property '{column_name}' to an entity: {str(e)}")
                logger.info(f"Skipping property '{column_name}' mapping to an entity")
                continue      
        
        # Check if any entity has no properties
        for entity_type, entity_info in final_property_entity_mapping.items():
            if not entity_info.get('properties'):
                logger.warning(f"Entity '{entity_type}' has no properties assigned")
        
        # Update the state with the final mapping
        state['property_entity_mapping'] = final_property_entity_mapping
        
    except Exception as e:
        logger.error(f"Error in property mapping: {str(e)}")
        state['error_messages'].append(f"Error in property mapping: {str(e)}")
        state['property_entity_mapping'] = {}
    
    return state
