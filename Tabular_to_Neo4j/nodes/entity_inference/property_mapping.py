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
from Tabular_to_Neo4j.utils.metadata_utils import load_metadata_for_csv, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows
from Tabular_to_Neo4j.utils.analytics_utils import analyze_column
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS

# Configure logging
logger = get_logger(__name__)


def map_properties_to_entities_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Map properties to their respective entities and create a clear property-entity mapping.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with property_entity_mapping
    """
    logger.info("Starting property-to-entity mapping process")
    
    # Initialize error messages list if it doesn't exist
    if 'error_messages' not in state:
        state['error_messages'] = []
    
    try:
        # Get the reconciled entity-property classification
        classification = state.get('entity_property_consensus', {})
        if not classification:
            error_msg = "Cannot map properties to entities: missing reconciled classification"
            logger.error(error_msg)
            state['error_messages'].append(error_msg)
            return state
        
        # Get the metadata if available
        metadata = None
        if 'csv_file_path' in state:
            csv_file_path = state['csv_file_path']
            metadata = load_metadata_for_csv(csv_file_path)
            
        # Get the column analytics
        column_analytics = state.get('column_analytics', {})
        
        # Separate entities and properties
        entities = []
        properties = []
        
        for column_name, info in classification.items():
            column_class = info.get('classification', '')
            
            if column_class == 'entity':
                entities.append(column_name)
                logger.debug(f"Identified entity column: {column_name}")
            
            elif column_class == 'property':
                properties.append(column_name)
                logger.debug(f"Identified property column: {column_name}")
        
        logger.info(f"Found {len(entities)} entities and {len(properties)} properties")
        
        # If no entities or properties, return the state unchanged
        if not entities:
            logger.warning("No entities found, skipping property mapping")
            return state
            
        # Use the first entity as our main entity
        main_entity = entities[0]
        logger.info(f"Using {main_entity} as the main entity")
        
        # Create a simple property-entity mapping
        property_entity_mapping = {
            main_entity: {
                'type': 'entity',
                'properties': []
            }
        }
        
        # Add all properties to the main entity
        for prop in properties:
            property_key = to_neo4j_property_name(prop)
            property_entity_mapping[main_entity]['properties'].append({
                'column_name': prop,
                'property_key': property_key
            })
            logger.info(f"Mapped property '{prop}' to entity '{main_entity}'")
        
        # Update the state with the final mapping
        state['property_entity_mapping'] = property_entity_mapping
        
    except Exception as e:
        logger.error(f"Error in property mapping: {str(e)}")
        state['error_messages'].append(f"Error in property mapping: {str(e)}")
        state['property_entity_mapping'] = {}
    
    return state
