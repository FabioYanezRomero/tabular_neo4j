"""
Relationship inference module for entity inference in the Tabular to Neo4j converter.
This module handles inferring relationships between entity types.
"""

from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def infer_entity_relationships_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Infer relationships between entity types.
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
        
        # Extract entity types
        entity_types = [
            item['entity_type'] 
            for key, item in mapping.items() 
            if item.get('type') == 'entity'
        ]
        
        logger.info(f"Found {len(entity_types)} entity types for relationship inference: {', '.join(entity_types)}")
        
        if not entity_types:
            logger.warning("No entity types found for relationship inference")
            state['entity_relationships'] = []
            return state
        
        # Extract existing relationships
        existing_relationships = [
            item.get('relationship', {})
            for key, item in mapping.items()
            if item.get('type') == 'property' and item.get('relationship')
        ]
        
        # Remove any empty relationships
        existing_relationships = [r for r in existing_relationships if r]
        
        logger.info(f"Found {len(entity_types)} entity types and {len(existing_relationships)} existing relationships")
        
        # If we only have one entity type, there are no relationships to infer
        if len(entity_types) <= 1:
            logger.info("Only one entity type found, no relationships to infer")
            state['entity_relationships'] = []
            return state
        
        # Prepare entity information for the LLM
        entity_info = []
        for entity_type in entity_types:
            # Get properties for this entity
            properties = [
                {
                    'name': key,
                    'property_name': item.get('property_name', key),
                    'data_type': item.get('data_type', 'unknown'),
                    'semantic_type': item.get('semantic_type', 'unknown'),
                    'uniqueness': item.get('uniqueness', 0)
                }
                for key, item in mapping.items()
                if item.get('type') == 'property' and item.get('entity_type') == entity_type
            ]
            
            entity_info.append({
                'entity_type': entity_type,
                'properties': properties
            })
        
        # Look for consensus in existing relationships
        consensus = {}
        for rel in existing_relationships:
            source = rel.get('source_entity')
            target = rel.get('target_entity')
            rel_type = rel.get('relationship_type')
            
            if source and target and rel_type:
                key = f"{source}_{rel_type}_{target}"
                if key not in consensus:
                    consensus[key] = {
                        'source_entity': source,
                        'target_entity': target,
                        'relationship_type': rel_type,
                        'count': 0,
                        'source_columns': []
                    }
                
                consensus[key]['count'] += 1
                if 'source_column' in rel:
                    consensus[key]['source_columns'].append(rel['source_column'])
        
        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
        
        # Format the prompt with entity information and metadata
        prompt = format_prompt('infer_entity_relationships.txt',
                              entity_info=str(entity_info),
                              existing_relationships=str(existing_relationships),
                              consensus_data=str(consensus),
                              metadata_text=metadata_text)
        
        # Call the LLM for relationship inference
        logger.debug("Calling LLM for relationship inference")
        try:
            response = call_llm_with_json_output(prompt, state_name="infer_entity_relationships")
            
            # Extract the inferred relationships
            inferred_relationships = response.get('relationships', [])
            
            if not inferred_relationships and not existing_relationships:
                logger.warning("No relationships inferred by LLM and no existing relationships found")
                
                # Create default relationships between all entity types
                if len(entity_types) > 1:
                    # Use the first entity as a source for simplicity
                    source_entity = entity_types[0]
                    logger.info(f"Creating default relationships from {source_entity} to other entities")
                    
                    for target_entity in entity_types[1:]:
                        relationship_type = f"RELATED_TO_{target_entity.upper()}"
                        inferred_relationships.append({
                            'source_entity': source_entity,
                            'relationship_type': relationship_type,
                            'target_entity': target_entity,
                            'properties': [],
                            'cardinality': "MANY_TO_MANY"
                        })
                        
                        logger.info(f"Created default relationship: {source_entity} {relationship_type} {target_entity}")
            
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
            logger.error(f"LLM relationship inference failed: {str(e)}")
            
            # If LLM call fails, create default relationships
            if not existing_relationships:
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
                        
                        logger.info(f"Created default relationship after error: {primary_entity} HAS_{entity_type.upper()} {entity_type}")
                
                state['entity_relationships'] = default_relationships
            else:
                # Use existing relationships if available
                state['entity_relationships'] = existing_relationships
                
            state['error_messages'].append(f"LLM relationship inference failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error inferring entity relationships: {str(e)}")
        state['error_messages'].append(f"Error inferring entity relationships: {str(e)}")
        state['entity_relationships'] = []
        
    return state
