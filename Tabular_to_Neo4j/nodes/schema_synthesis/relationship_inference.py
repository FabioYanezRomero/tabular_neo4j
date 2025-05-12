"""
Relationship inference module for schema synthesis in the Tabular to Neo4j converter.
This module handles the fourth step in schema synthesis: inferring relationships between entity types.
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

def infer_entity_relationships_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Fourth step in schema synthesis: Infer relationships between entity types.
    Uses LLM to identify and characterize relationships between entities.
    
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
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Extract entity types
        entity_types = [
            item['entity_type'] 
            for key, item in mapping.items() 
            if item.get('type') == 'entity'
        ]
        
        # Extract existing relationships
        existing_relationships = [
            {
                'source_entity': item['source_entity'],
                'relationship_type': item['relationship_type'],
                'target_entity': item['target_entity'],
                'properties': item['properties']
            }
            for key, item in mapping.items() 
            if item.get('type') == 'relationship'
        ]
        
        logger.debug(f"Found {len(entity_types)} entity types and {len(existing_relationships)} existing relationships")
        
        # If we only have one entity type, there are no relationships to infer
        if len(entity_types) <= 1:
            logger.info("Only one entity type found, no relationships to infer")
            state['entity_relationships'] = existing_relationships
            return state
        
        # Get consensus data for additional context
        consensus = state.get('entity_property_consensus', {})
        
        # Prepare entity info for the prompt
        entity_info = []
        for entity_type in entity_types:
            entity_data = next(
                (item for key, item in mapping.items() if item.get('type') == 'entity' and item.get('entity_type') == entity_type),
                None
            )
            
            if not entity_data:
                continue
                
            # Get properties for this entity
            properties = entity_data.get('properties', [])
            
            # Find columns that created this entity type (if any)
            source_columns = [
                col_name for col_name, info in consensus.items()
                if info.get('classification') == 'new_entity_type' and info.get('entity_type') == entity_type
            ]
            
            entity_info.append({
                'entity_type': entity_type,
                'is_primary': entity_data.get('is_primary', False),
                'properties': [prop.get('property_key') for prop in properties],
                'source_columns': source_columns
            })
        
        # Format the prompt with entity information
        prompt = format_prompt('infer_entity_relationships.txt',
                              entity_info=str(entity_info),
                              primary_entity=primary_entity,
                              existing_relationships=str(existing_relationships),
                              consensus_data=str(consensus))
        
        # Call the LLM for relationship inference
        logger.debug("Calling LLM for relationship inference")
        try:
            response = call_llm_with_json_output(prompt, state_name="infer_entity_relationships")
            
            # Extract the inferred relationships
            inferred_relationships = response.get('relationships', [])
            
            if not inferred_relationships and not existing_relationships:
                logger.warning("No relationships inferred by LLM and no existing relationships found")
                
                # Create default relationships from primary entity to other entities
                for entity_type in entity_types:
                    if entity_type != primary_entity:
                        inferred_relationships.append({
                            'source_entity': primary_entity,
                            'relationship_type': f"HAS_{entity_type.upper()}",
                            'target_entity': entity_type,
                            'properties': [],
                            'cardinality': "ONE_TO_MANY"
                        })
                        
                        logger.info(f"Created default relationship: {primary_entity} HAS_{entity_type.upper()} {entity_type}")
            
            # Combine with existing relationships
            all_relationships = existing_relationships.copy()
            
            # Add new inferred relationships, avoiding duplicates
            for rel in inferred_relationships:
                # Check if this relationship already exists
                exists = any(
                    existing['source_entity'] == rel['source_entity'] and
                    existing['relationship_type'] == rel['relationship_type'] and
                    existing['target_entity'] == rel['target_entity']
                    for existing in all_relationships
                )
                
                if not exists:
                    all_relationships.append(rel)
                    logger.info(f"Added inferred relationship: {rel['source_entity']} {rel['relationship_type']} {rel['target_entity']}")
            
            # Update the state
            state['entity_relationships'] = all_relationships
            
        except Exception as e:
            logger.error(f"Error inferring entity relationships: {str(e)}")
            
            # Use existing relationships as fallback
            if existing_relationships:
                logger.info(f"Using {len(existing_relationships)} existing relationships as fallback")
                state['entity_relationships'] = existing_relationships
            else:
                # Create default relationships from primary entity to other entities
                default_relationships = []
                for entity_type in entity_types:
                    if entity_type != primary_entity:
                        default_relationships.append({
                            'source_entity': primary_entity,
                            'relationship_type': f"HAS_{entity_type.upper()}",
                            'target_entity': entity_type,
                            'properties': [],
                            'cardinality': "ONE_TO_MANY"
                        })
                
                logger.info(f"Created {len(default_relationships)} default relationships as fallback")
                state['entity_relationships'] = default_relationships
            
            state['error_messages'].append(f"Error inferring entity relationships: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in relationship inference process: {str(e)}")
        state['error_messages'].append(f"Error in relationship inference process: {str(e)}")
        state['entity_relationships'] = []
    
    return state
