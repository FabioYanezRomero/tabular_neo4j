"""
Schema synthesis nodes for the LangGraph CSV analysis pipeline.
These nodes handle the process of transforming column analysis into Neo4j schema components.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def classify_entities_properties_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    First step in schema synthesis: Classify columns as entities or properties based on analytics and semantics.
    Uses LLM to classify each column based on its analytics and semantics.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with entity_property_classification
    """
    logger.info("Starting entity-property classification process")
    
    # Validate required inputs
    missing_inputs = []
    if state.get('column_analytics') is None:
        missing_inputs.append("column_analytics")
    if state.get('llm_column_semantics') is None:
        missing_inputs.append("llm_column_semantics")
    if state.get('final_header') is None:
        missing_inputs.append("final_header")
    
    if missing_inputs:
        error_msg = f"Cannot classify entities/properties: missing required analysis data: {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    # Log input data stats
    header_count = len(state['final_header'])
    analytics_count = len(state.get('column_analytics', {}))
    semantics_count = len(state.get('llm_column_semantics', {}))
    logger.debug(f"Input data: {header_count} columns, {analytics_count} analytics entries, {semantics_count} semantic entries")
    
    try:
        logger.debug("Initializing classification process")
        # Initialize classification dictionary
        classification = {}
        
        # Get primary entity label from filename
        csv_path = state['csv_file_path']
        primary_entity_label = get_primary_entity_from_filename(csv_path)
        file_name = os.path.basename(csv_path)
        logger.info(f"Derived primary entity label '{primary_entity_label}' from file: {file_name}")
        
        # Process each column using LLM for classification
        logger.info(f"Beginning classification of {len(state['final_header'])} columns")
        processed_columns = 0
        skipped_columns = 0
        
        for column_name in state['final_header']:
            logger.debug(f"Processing column: '{column_name}'")
            
            # Get analytics and semantics for this column
            analytics = state.get('column_analytics', {}).get(column_name, {})
            semantics = state.get('llm_column_semantics', {}).get(column_name, {})
            
            # Skip if we don't have both analytics and semantics
            if not analytics:
                logger.warning(f"Missing analytics for column '{column_name}', skipping")
                skipped_columns += 1
                continue
            if not semantics:
                logger.warning(f"Missing semantics for column '{column_name}', skipping")
                skipped_columns += 1
                continue
            
            # Log key analytics for debugging
            uniqueness = analytics.get('uniqueness', 0)
            null_percentage = analytics.get('null_percentage', 0)
            logger.debug(f"Column '{column_name}' analytics: uniqueness={uniqueness:.2f}, null_percentage={null_percentage:.2f}")
            
            # Get sample values for this column
            sample_values = []
            if state.get('processed_dataframe') is not None:
                df = state['processed_dataframe']
                # TODO: droping the null values this way might be a problem
                non_null_values = df[column_name].dropna()
                
                if len(non_null_values) > 0:
                    sample_count = min(5, len(non_null_values))
                    sample_values = non_null_values.sample(sample_count).tolist()
                    logger.debug(f"Sampled {len(sample_values)} values from column '{column_name}'")
                else:
                    logger.warning(f"Column '{column_name}' has no non-null values to sample")
            else:
                logger.warning("No processed dataframe available for sampling values")
            
            processed_columns += 1
            
            # Format the prompt with column information
            prompt = format_prompt('classify_entities_properties.txt',
                                 file_name=file_name,
                                 column_name=column_name,
                                 primary_entity=primary_entity_label,
                                 sample_values=str(sample_values),
                                 uniqueness_ratio=analytics.get('uniqueness_ratio', 0),
                                 cardinality=analytics.get('cardinality', 0),
                                 data_type=analytics.get('data_type', 'unknown'),
                                 missing_percentage=analytics.get('missing_percentage', 0) * 100,
                                 semantic_type=semantics.get('semantic_type', 'Unknown'),
                                 llm_role=semantics.get('neo4j_role', 'UNKNOWN'))
            
            # Call the LLM for entity/property classification
            logger.debug(f"Calling LLM for entity/property classification of column '{column_name}'")
            try:
                response = call_llm_with_json_output(prompt, state_name="classify_entities_properties")
                logger.debug(f"Received LLM classification response for '{column_name}'")
                
                # Extract the classification results
                classification_result = response.get('classification', 'entity_property')
                entity_type = response.get('entity_type', primary_entity_label)
                relationship = response.get('relationship_to_primary', '')
                
                classification[column_name] = {
                    'column_name': column_name,
                    'classification': classification_result,
                    'entity_type': entity_type,
                    'relationship_to_primary': relationship,
                    'property_name': response.get('property_name', column_name),
                    'reasoning': response.get('reasoning', ''),
                    'analytics': analytics,
                    'semantics': semantics
                }
                
                logger.info(f"Classified '{column_name}' as '{classification_result}' with entity type '{entity_type}'")
                if relationship:
                    logger.debug(f"Relationship to primary entity: '{relationship}'")
                    
            # TODO: check and improve the fallback evaluation whenever the model fails
            except Exception as e:
                logger.error(f"LLM entity classification failed for '{column_name}': {str(e)}")
                
                # If LLM call fails, use a fallback classification based on analytics
                uniqueness = analytics.get('uniqueness', 0)
                is_unique_enough = uniqueness > UNIQUENESS_THRESHOLD
                fallback_classification = 'entity' if is_unique_enough else 'property'
                
                logger.warning(f"Using fallback classification for '{column_name}': {fallback_classification} based on uniqueness ({uniqueness:.2f})")
                
                classification[column_name] = {
                    'column_name': column_name,
                    'classification': fallback_classification,
                    'entity_type': primary_entity_label if is_unique_enough else '',
                    'relationship_to_primary': 'IS_A' if is_unique_enough else '',
                    'property_name': column_name,
                    'reasoning': f'Fallback classification based on uniqueness ({uniqueness:.2f})',
                    'analytics': analytics,
                    'semantics': semantics
                }
                
                state['error_messages'].append(f"LLM entity classification failed for {column_name}: {str(e)}")
                
            # Fallback classification based on the existing semantics
            neo4j_role = semantics.get('neo4j_role', 'UNKNOWN')
            
            if neo4j_role == 'PRIMARY_ENTITY_IDENTIFIER':
                classification[column_name] = {
                    'column_name': column_name,
                    'classification': 'entity_identifier',
                    'entity_type': primary_entity_label,
                    'uniqueness_ratio': analytics.get('uniqueness_ratio', 0),
                    'semantic_type': semantics.get('semantic_type', 'Unknown'),
                    'neo4j_property_key': to_neo4j_property_name(column_name),
                    'note': 'Fallback classification due to LLM error'
                }
            elif neo4j_role == 'NEW_NODE_TYPE_VALUES':
                new_node_label = semantics.get('new_node_label_suggestion', column_name.capitalize())
                relationship_type = semantics.get('relationship_type_suggestion', f"HAS_{new_node_label.upper()}")
                
                classification[column_name] = {
                    'column_name': column_name,
                    'classification': 'new_entity_type',
                    'entity_type': new_node_label,
                    'relationship_to_primary': relationship_type,
                    'primary_entity': primary_entity_label,
                    'semantic_type': semantics.get('semantic_type', 'Unknown'),
                    'neo4j_property_key': 'name',
                    'note': 'Fallback classification due to LLM error'
                }
            else:
                # Default to property if no specific role matches
                # TODO: check and improve the fallback evaluation whenever the model fails
                classification[column_name] = {
                    'column_name': column_name,
                    'classification': 'entity_property',
                    'entity_type': primary_entity_label,
                    'semantic_type': semantics.get('semantic_type', 'Unknown'),
                    'neo4j_property_key': to_neo4j_property_name(column_name),
                    'note': 'Fallback classification due to LLM error'
                }
        
        # Update the state
        state['entity_property_classification'] = classification
        
    except Exception as e:
        logger.error(f"Error classifying entities/properties: {str(e)}")
        state['error_messages'].append(f"Error classifying entities/properties: {str(e)}")
        state['entity_property_classification'] = {}
    
    return state

def reconcile_entity_property_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Second step in schema synthesis: Reconcile analytics-based and LLM-based classifications
    to create a consensus model of entities and properties.
    Uses LLM to reconcile different classification approaches.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with entity_property_consensus
    """
    logger.info("Starting entity-property reconciliation process")
    
    # Validate required inputs
    if state.get('entity_property_classification') is None:
        error_msg = "Cannot reconcile entity/property classifications: missing classification data"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Get the initial classification
        classification = state['entity_property_classification']
        file_name = os.path.basename(state['csv_file_path'])
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Initialize consensus dictionary
        consensus = {}
        
        # Track entity types for later use
        entity_types = set([primary_entity])
        
        # First pass: identify all entity types
        for column_name, info in classification.items():
            if info['classification'] in ['new_entity_type']:
                entity_types.add(info['entity_type'])
        
        # Process each column using LLM for reconciliation
        for column_name, info in classification.items():
            # Get analytics and semantics for this column
            analytics = state.get('column_analytics', {}).get(column_name, {})
            semantics = state.get('llm_column_semantics', {}).get(column_name, {})
            
            # Skip if we don't have both analytics and semantics
            if not analytics or not semantics:
                consensus[column_name] = info  # Use initial classification as fallback
                continue
            
            # Get sample values for this column
            sample_values = []
            if state.get('processed_dataframe') is not None:
                sample_values = state['processed_dataframe'][column_name].dropna().sample(
                    min(5, len(state['processed_dataframe'][column_name].dropna()))
                ).tolist() if len(state['processed_dataframe'][column_name].dropna()) > 0 else []
            
            # Determine analytics-based classification
            analytics_classification = "entity_property"  # Default
            uniqueness = analytics.get('uniqueness', 0)
            cardinality = analytics.get('cardinality', 0)
            
            if uniqueness > UNIQUENESS_THRESHOLD:
                analytics_classification = "entity_identifier"
                logger.debug(f"Analytics suggests '{column_name}' is an entity_identifier (uniqueness: {uniqueness:.2f})")

            elif analytics.get('cardinality', 0) < len(state['processed_dataframe']) * 0.1 and analytics.get('cardinality', 0) > 1:
                analytics_classification = "new_entity_type"
            
            # Format the prompt with column information
            prompt = format_prompt('reconcile_entity_property.txt',
                                  file_name=file_name,
                                  column_name=column_name,
                                  primary_entity=primary_entity,
                                  initial_classification=str(info),
                                  analytics_classification=analytics_classification,
                                  llm_classification=semantics.get('neo4j_role', 'UNKNOWN'),
                                  sample_values=str(sample_values),
                                  entity_types=str(list(entity_types)))
            
            # Call the LLM for reconciliation
            try:
                response = call_llm_with_json_output(prompt, state_name="reconcile_entity_property")
                
                # Extract the reconciliation results
                consensus_classification = response.get('consensus_classification', info['classification'])
                entity_type = response.get('entity_type', info.get('entity_type', primary_entity))
                relationship_to_primary = response.get('relationship_to_primary', info.get('relationship_to_primary', ''))
                neo4j_property_key = response.get('neo4j_property_key', info.get('neo4j_property_key', to_neo4j_property_name(column_name)))
                confidence = response.get('confidence', 0.5)
                reasoning = response.get('reasoning', '')
                
                # Create consensus entry
                consensus[column_name] = {
                    'column_name': column_name,
                    'classification': consensus_classification,
                    'entity_type': entity_type,
                    'relationship_to_primary': relationship_to_primary,
                    'neo4j_property_key': neo4j_property_key,
                    'semantic_type': info.get('semantic_type', 'Unknown'),
                    'confidence': confidence,
                    'reasoning': reasoning
                }
                
                # Add additional fields based on classification type
                if consensus_classification == 'entity_identifier':
                    consensus[column_name]['uniqueness_ratio'] = analytics.get('uniqueness_ratio', 0)
                
                elif consensus_classification == 'new_entity_type':
                    # Ensure we have a relationship type
                    if not relationship_to_primary:
                        consensus[column_name]['relationship_to_primary'] = f"HAS_{entity_type.upper()}"
                    
                    consensus[column_name]['primary_entity'] = primary_entity
                
                # Add entity type to our set if it's a new one
                if consensus_classification == 'new_entity_type':
                    entity_types.add(entity_type)
                
            except Exception as e:
                logger.error(f"Error reconciling column {column_name}: {str(e)}")
                consensus[column_name] = info  # Use initial classification as fallback
        
        # Second pass: resolve secondary entity properties
        for column_name, info in consensus.items():
            if info['classification'] == 'secondary_entity_property':
                # Try to find the associated entity type
                associated_entity = find_associated_entity_type(
                    column_name, 
                    info,
                    consensus,
                    entity_types,
                    state['llm_column_semantics']
                )
                
                if associated_entity:
                    info['entity_type'] = associated_entity
                else:
                    # If we can't find an association, default to primary entity
                    info['entity_type'] = primary_entity
                    info['note'] = "No clear entity association found, defaulted to primary entity"
        
        # Third pass: resolve relationship properties
        for column_name, info in consensus.items():
            if info['classification'] == 'relationship_property':
                # Try to find the associated relationship
                relationship_info = find_associated_relationship(
                    column_name,
                    info,
                    consensus
                )
                
                if relationship_info:
                    info['relationship_type'] = relationship_info['relationship']
                    info['source_entity'] = relationship_info['source_entity']
                    info['target_entity'] = relationship_info['target_entity']
                else:
                    # If we can't find an association, convert to primary entity property
                    info['classification'] = 'entity_property'
                    info['entity_type'] = primary_entity
                    info['note'] = "Originally classified as relationship property but no association found"
        
        # Update the state
        state['entity_property_consensus'] = consensus
        
    except Exception as e:
        logger.error(f"Error reconciling entity/property classifications: {str(e)}")
        state['error_messages'].append(f"Error reconciling entity/property classifications: {str(e)}")
        state['entity_property_consensus'] = state.get('entity_property_classification', {})
    
    return state

# TODO: This node is way too complex, maybe we should simplify it
def map_properties_to_entities_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Third step in schema synthesis: Map properties to their respective entities
    and create a clear property-entity mapping.
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
    
    # Log input data stats
    consensus_count = len(state.get('entity_property_consensus', {}))
    logger.info(f"Mapping {consensus_count} properties to their respective entities")
    
    # TODO: check if the concept of primary entity makes sense
    try:
        # Get the consensus classification
        consensus = state['entity_property_consensus']
        csv_path = state['csv_file_path']
        file_name = os.path.basename(csv_path)
        primary_entity = get_primary_entity_from_filename(csv_path)
        logger.info(f"Using primary entity '{primary_entity}' from file: {file_name}")
        
        # Create a set of entity types
        entity_types = set([primary_entity])
        
        # First pass: identify all entity types
        logger.debug("Identifying all entity types from consensus data")
        for column_name, info in consensus.items():
            if info['classification'] in ['entity_identifier', 'entity_property', 'new_entity_type']:
                entity_type = info['entity_type']
                entity_types.add(entity_type)
                logger.debug(f"Found entity type '{entity_type}' from column '{column_name}'")
        
        logger.info(f"Identified {len(entity_types)} entity types: {', '.join(entity_types)}")

        
        # Prepare property classifications for the LLM
        logger.debug("Preparing property classifications for LLM input")
        property_classifications = {}
        for column_name, info in consensus.items():
            classification = info['classification']
            entity_type = info.get('entity_type', primary_entity)
            semantic_type = info.get('semantic_type', 'Unknown')
            
            property_classifications[column_name] = {
                'classification': classification,
                'entity_type': entity_type,
                'semantic_type': semantic_type
            }
            logger.debug(f"Column '{column_name}' classified as '{classification}' with entity type '{entity_type}'")
        
        # Format the prompt with entity and property information
        logger.debug("Formatting prompt for property-entity mapping")
        prompt = format_prompt('map_properties_to_entities.txt',
                             file_name=file_name,
                             primary_entity=primary_entity,
                             entity_types=str(list(entity_types)),
                             property_classifications=str(property_classifications),
                             entity_property_consensus=str(consensus),
                             column_semantics=str(state.get('llm_column_semantics', {})))
        
        # Call the LLM for property-entity mapping
        logger.info("Calling LLM for property-entity mapping")
        try:
            response = call_llm_with_json_output(prompt, state_name="map_properties_to_entities")
            logger.debug("Received LLM response for property-entity mapping")
            
            # Extract the mapping results
            property_entity_map = response.get('property_entity_mapping', {})
            entity_properties = response.get('entity_properties', {})
            reasoning = response.get('reasoning', '')
            
            if not property_entity_map:
                logger.warning("LLM returned empty property-entity mapping, will use fallbacks")
            else:
                logger.info(f"LLM mapped {len(property_entity_map)} properties to entities")
            
            # Validate and fill in any missing mappings
            logger.debug("Validating property-entity mappings and filling gaps")
            missing_mappings = 0
            

            # TODO: Check if the secondary_entity_property classification makes sense, i think an entity can have more than two properties and neither of those should be primary, at least at this point
            for column_name, info in consensus.items():
                if column_name not in property_entity_map:
                    missing_mappings += 1
                    # Default mapping based on consensus
                    # TODO: I think we shouldn't consider entity identifiers here, as in the future when the database grows those dentifiers might be inappropiate
                    if info['classification'] in ['entity_property', 'secondary_entity_property']:
                        entity_type = info.get('entity_type', primary_entity)
                        property_entity_map[column_name] = entity_type
                        logger.warning(f"Missing mapping for '{column_name}', defaulting to entity type '{entity_type}'")
                    elif info['classification'] == 'new_entity_type':
                        entity_type = info.get('entity_type', column_name.capitalize())
                        property_entity_map[column_name] = entity_type
                        logger.warning(f"Missing mapping for '{column_name}', defaulting to new entity type '{entity_type}'")
                else:
                    logger.debug(f"Column '{column_name}' mapped to entity '{property_entity_map[column_name]}'")
            
            if missing_mappings > 0:
                logger.warning(f"Added {missing_mappings} missing mappings using default rules")

            
            # Ensure all entity types have an entry in entity_properties
            logger.debug("Ensuring all entity types have entries in entity_properties")
            for entity_type in entity_types:
                if entity_type not in entity_properties:
                    logger.debug(f"Creating empty property list for entity type '{entity_type}'")
                    entity_properties[entity_type] = []
            
            # Validate entity_properties structure and fill in any missing properties
            logger.debug("Validating entity_properties structure and filling gaps")
            properties_added = 0
            
            for column_name, entity_type in property_entity_map.items():
                if entity_type in entity_properties:
                    # Check if this property is already in the entity's properties
                    if not any(prop.get('column_name') == column_name for prop in entity_properties[entity_type]):
                        # Add it with default values
                        info = consensus.get(column_name, {})
                        neo4j_property_key = info.get('neo4j_property_key', to_neo4j_property_name(column_name))
                        semantic_type = info.get('semantic_type', 'Unknown')
                        is_identifier = info.get('classification') == 'entity_identifier'
                        
                        entity_properties[entity_type].append({
                            'column_name': column_name,
                            'neo4j_property_key': neo4j_property_key,
                            'semantic_type': semantic_type,
                            'is_identifier': is_identifier
                        })
                        
                        properties_added += 1
                        logger.debug(f"Added property '{column_name}' to entity '{entity_type}' with key '{neo4j_property_key}'")
                else:
                    logger.warning(f"Entity type '{entity_type}' not found in entity_properties for column '{column_name}'")
            
            if properties_added > 0:
                logger.info(f"Added {properties_added} missing properties to entity_properties")
            
            # Summarize the mapping results
            for entity_type, props in entity_properties.items():
                identifier_props = [p['column_name'] for p in props if p.get('is_identifier', False)]
                regular_props = [p['column_name'] for p in props if not p.get('is_identifier', False)]
                
                if identifier_props:
                    logger.info(f"Entity '{entity_type}' identifiers: {', '.join(identifier_props)}")
                if regular_props:
                    logger.info(f"Entity '{entity_type}' properties: {', '.join(regular_props)}")
            
            # Update the state with the mapping results
            logger.info("Property-entity mapping completed successfully")
            
        except Exception as e:
            logger.error(f"Error mapping properties to entities: {str(e)}", exc_info=True)
            state['error_messages'].append(f"Error mapping properties to entities: {str(e)}")
            
            # Create fallback mapping based on consensus
            logger.warning("Creating fallback property-entity mapping based on consensus")
            property_entity_map = {}
            entity_properties = {}
            
            # Add all entity types
            for entity_type in entity_types:
                entity_properties[entity_type] = []
                logger.debug(f"Created empty property list for entity type '{entity_type}'")
            
            # Map each property based on consensus
            mapped_count = 0

            # TODO: I think is identifier should not be used, we can store a representative field as property, but the unique Id should be handled by the database
            for column_name, info in consensus.items():
                if info['classification'] in ['entity_identifier', 'entity_property', 'secondary_entity_property']:
                    entity_type = info.get('entity_type', primary_entity)
                    property_entity_map[column_name] = entity_type
                    
                    # Add to entity properties
                    neo4j_property_key = info.get('neo4j_property_key', to_neo4j_property_name(column_name))
                    is_identifier = info.get('classification') == 'entity_identifier'
                    
                    entity_properties[entity_type].append({
                        'column_name': column_name,
                        'neo4j_property_key': neo4j_property_key,
                        'semantic_type': info.get('semantic_type', 'Unknown'),
                        'is_identifier': is_identifier
                    })
                    
                    mapped_count += 1
                    logger.debug(f"Fallback: Mapped '{column_name}' to entity '{entity_type}' as {'identifier' if is_identifier else 'property'}")
                    
                elif info['classification'] == 'new_entity_type':
                    entity_type = info.get('entity_type', column_name.capitalize())
                    property_entity_map[column_name] = entity_type
                    
                    # Add to entity properties if entity type exists
                    if entity_type in entity_properties:
                        entity_properties[entity_type].append({
                            'column_name': column_name,
                            'neo4j_property_key': 'name',  # Default for new entity types
                            'semantic_type': info.get('semantic_type', 'Unknown'),
                            'is_identifier': True  # Default for new entity types
                        })
                        
                        mapped_count += 1
                        logger.debug(f"Fallback: Mapped '{column_name}' to new entity '{entity_type}' as identifier")
            
            logger.info(f"Fallback mapping created with {mapped_count} properties mapped to entities")
        
        # Update the state
        state['property_entity_mapping'] = property_entity_map
        state['entity_properties'] = entity_properties
        
    except Exception as e:
        logger.error(f"Error mapping properties to entities: {str(e)}")
        state['error_messages'].append(f"Error mapping properties to entities: {str(e)}")
        state['property_entity_mapping'] = {}
        state['entity_properties'] = {}
    
    return state

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
    missing_inputs = []
    if state.get('entity_property_consensus') is None:
        missing_inputs.append("entity_property_consensus")
    if state.get('property_entity_mapping') is None:
        missing_inputs.append("property_entity_mapping")
    if state.get('entity_properties') is None:
        missing_inputs.append("entity_properties")
    
    if missing_inputs:
        error_msg = f"Cannot infer entity relationships: missing {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    # Log input data stats
    property_count = len(state.get('property_entity_mapping', {}))
    entity_count = len(state.get('entity_properties', {}))
    logger.info(f"Inferring relationships between {entity_count} entities with {property_count} mapped properties")
    
    try:
        # Get the necessary data from state
        consensus = state['entity_property_consensus']
        property_entity_map = state['property_entity_mapping']
        entity_properties = state['entity_properties']
        csv_path = state['csv_file_path']
        file_name = os.path.basename(csv_path)
        primary_entity = get_primary_entity_from_filename(csv_path)
        logger.info(f"Using primary entity '{primary_entity}' from file: {file_name}")
        
        # Get all entity types
        entity_types = list(entity_properties.keys())
        logger.debug(f"Entity types for relationship inference: {entity_types}")

        # Format the prompt with entity and relationship information
        logger.debug("Formatting prompt for entity relationship inference")
        prompt = format_prompt('infer_entity_relationships.txt',
                             file_name=file_name,
                             primary_entity=primary_entity,
                             entity_types=str(list(entity_types)),
                             entity_property_consensus=str(consensus),
                             property_entity_mapping=str(property_entity_map),
                             column_semantics=str(state.get('llm_column_semantics', {})))
        
        # Call the LLM for entity relationship inference
        logger.info("Calling LLM for entity relationship inference")
        try:
            response = call_llm_with_json_output(prompt, state_name="infer_entity_relationships")
            logger.debug("Received LLM response for entity relationship inference")
            
            # Extract the relationship results
            relationships = response.get('entity_relationships', [])
            reasoning = response.get('reasoning', '')
            
            if not relationships:
                logger.warning("LLM returned no entity relationships")
            else:
                logger.info(f"LLM inferred {len(relationships)} entity relationships")
                for i, rel in enumerate(relationships):
                    if all(k in rel for k in ['source_entity', 'target_entity', 'relationship_type']):
                        logger.debug(f"Relationship {i+1}: ({rel['source_entity']})-[:{rel['relationship_type']}]->({rel['target_entity']})")
            
            # Validate relationships
            logger.debug("Validating inferred relationships")
            validated_relationships = []
            skipped_count = 0
            
            for rel in relationships:
                # Ensure required fields are present
                if 'source_entity' not in rel or 'target_entity' not in rel or 'relationship_type' not in rel:
                    logger.warning(f"Skipping incomplete relationship: {rel}")
                    skipped_count += 1
                    continue
                
                # Ensure entities exist in our entity types
                if rel['source_entity'] not in entity_types or rel['target_entity'] not in entity_types:
                    logger.warning(f"Skipping relationship with unknown entity: {rel['source_entity']} or {rel['target_entity']}")
                    skipped_count += 1
                    continue
                
                # Ensure properties are properly formatted
                property_count = 0
                if 'properties' in rel and isinstance(rel['properties'], list):
                    validated_properties = []
                    for prop in rel['properties']:
                        if 'column_name' in prop and prop['column_name'] in consensus:
                            # Add neo4j_property_key if missing
                            if 'neo4j_property_key' not in prop:
                                prop_key = to_neo4j_property_name(prop['column_name'])
                                prop['neo4j_property_key'] = prop_key
                                logger.debug(f"Added missing neo4j_property_key '{prop_key}' to relationship property '{prop['column_name']}'")
                            
                            # Add semantic_type if missing
                            if 'semantic_type' not in prop:
                                semantic_type = consensus[prop['column_name']].get('semantic_type', 'Unknown')
                                prop['semantic_type'] = semantic_type
                                logger.debug(f"Added missing semantic_type '{semantic_type}' to relationship property '{prop['column_name']}'")
                            
                            validated_properties.append(prop)
                            property_count += 1
                    
                    rel['properties'] = validated_properties
                else:
                    rel['properties'] = []
                
                logger.debug(f"Validated relationship ({rel['source_entity']})-[:{rel['relationship_type']}]->({rel['target_entity']}) with {property_count} properties")
                validated_relationships.append(rel)
            
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} invalid relationships")
            
            logger.info(f"Validated {len(validated_relationships)} entity relationships")

            
            # Log the relationship results
            logger.info(f"Inferred entity relationships: {validated_relationships}")
            
            # Update relationships with the validated ones
            relationships = validated_relationships
            
        except Exception as e:
            logger.error(f"Error calling LLM for entity relationship inference: {str(e)}")
            
            # Fallback: Create relationships manually based on consensus
            relationships = []
            
            # First, add relationships from columns classified as new entity types
            for column_name, info in consensus.items():
                if info['classification'] == 'new_entity_type':
                    relationship = {
                        'source_entity': primary_entity,
                        'target_entity': info['entity_type'],
                        'relationship_type': info['relationship_to_primary'],
                        'cardinality': 'ONE_TO_MANY',  # Default cardinality
                        'source_column': None,  # Primary entity doesn't come from a specific column
                        'target_column': column_name,
                        'properties': []  # Will be populated with relationship properties
                    }
                    relationships.append(relationship)
            
            # Then, add relationship properties
            for column_name, info in consensus.items():
                if info['classification'] == 'relationship_property':
                    # Find the relationship this property belongs to
                    for rel in relationships:
                        if (rel['source_entity'] == info.get('source_entity') and 
                            rel['target_entity'] == info.get('target_entity') and
                            rel['relationship_type'] == info.get('relationship_type')):
                            
                            # Add this property to the relationship
                            rel['properties'].append({
                                'column_name': column_name,
                                'neo4j_property_key': info['neo4j_property_key'],
                                'semantic_type': info['semantic_type']
                            })
                            break
        
        # Update the state
        state['entity_relationships'] = relationships
        
    except Exception as e:
        logger.error(f"Error inferring entity relationships: {str(e)}")
        state['error_messages'].append(f"Error inferring entity relationships: {str(e)}")
        state['entity_relationships'] = []
    
    return state

def generate_cypher_templates_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Fifth step in schema synthesis: Generate template Cypher queries for the inferred schema.
    Uses LLM to create Cypher query templates for loading and querying the Neo4j graph model.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with cypher_query_templates
    """
    if (state.get('entity_property_consensus') is None or 
        state.get('entity_relationships') is None or
        state.get('entity_properties') is None):
        state['error_messages'].append("Cannot generate Cypher templates: missing required data")
        return state
    
    try:
        # Get the necessary data from state
        consensus = state['entity_property_consensus']
        relationships = state['entity_relationships']
        entity_properties = state['entity_properties']
        file_name = os.path.basename(state['csv_file_path'])
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Get all entity types
        entity_types = list(entity_properties.keys())
        
        # Format the prompt with entity, property, and relationship information
        prompt = format_prompt('generate_cypher_templates.txt',
                             file_name=file_name,
                             primary_entity=primary_entity,
                             entity_types=str(entity_types),
                             entity_property_consensus=str(consensus),
                             entity_relationships=str(relationships),
                             entity_properties=str(entity_properties))
        
        # Call the LLM for Cypher template generation
        try:
            response = call_llm_with_json_output(prompt, state_name="generate_cypher_templates")
            
            # Extract the template results
            templates = response.get('cypher_templates', [])
            constraints_and_indexes = response.get('constraints_and_indexes', [])
            
            # Validate templates
            validated_templates = []
            for template in templates:
                # Ensure required fields are present
                if 'purpose' not in template or 'query' not in template:
                    logger.warning(f"Skipping incomplete template: {template}")
                    continue
                
                # Add description if missing
                if 'description' not in template:
                    template['description'] = template['purpose']
                
                validated_templates.append(template)
            
            # Add constraints and indexes to templates if any
            for constraint in constraints_and_indexes:
                if 'query' in constraint and 'type' in constraint:
                    validated_templates.append({
                        'purpose': f"{constraint['type']} on {constraint.get('entity_type', 'entity')}.{constraint.get('property', 'property')}",
                        'query': constraint['query'],
                        'description': f"Creates a {constraint['type'].lower()} for the schema"
                    })
            
            # Log the template results
            logger.info(f"Generated Cypher templates: {validated_templates}")
            
            # Update templates with the validated ones
            templates = validated_templates
            
        except Exception as e:
            logger.error(f"Error calling LLM for Cypher template generation: {str(e)}")
            
            # Fallback: Create templates manually
            templates = []
            
            # 1. Create template for creating primary entity nodes
            primary_props = entity_properties.get(primary_entity, [])
            if primary_props:
                # Find identifier properties
                identifiers = [p for p in primary_props if p.get('is_identifier', False)]
                
                # If no identifiers, use all properties
                if not identifiers:
                    identifiers = primary_props
                
                # Create MERGE query for primary entity
                identifier_props = ", ".join([f"{p['neo4j_property_key']}: row.{p['column_name']}" for p in identifiers])
                all_props = ", ".join([f"{p['neo4j_property_key']}: row.{p['column_name']}" for p in primary_props])
                
                query = f"""
                // Create primary {primary_entity} nodes
                LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row
                MERGE (e:{primary_entity} {{{identifier_props}}})
                ON CREATE SET e = {{{all_props}}}
                ON MATCH SET e = {{{all_props}}}
                """
                
                templates.append({
                    'purpose': f'Create {primary_entity} nodes',
                    'query': query.strip()
                })
            
            # 2. Create templates for secondary entity nodes and relationships
            for relationship in relationships:
                source_entity = relationship['source_entity']
                target_entity = relationship['target_entity']
                rel_type = relationship['relationship_type']
                target_column = relationship.get('target_column')
                
                if not target_column:
                    # Skip if we don't have a target column
                    continue
                
                # Get properties for the target entity
                target_props = entity_properties.get(target_entity, [])
                
                # Create MERGE query for secondary entity and relationship
                query = f"""
                // Create {target_entity} nodes and relationships to {source_entity}
                LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row
                MATCH (source:{source_entity})
                WHERE source.id = row.id  // Adjust this to match your primary key
                MERGE (target:{target_entity} {{name: row.{target_column}}})
                MERGE (source)-[r:{rel_type}]->(target)
                """
                
                # Add relationship properties if any
                if relationship.get('properties'):
                    props = ", ".join([f"r.{p['neo4j_property_key']} = row.{p['column_name']}" for p in relationship['properties']])
                    query += f"SET {props}"
                
                templates.append({
                    'purpose': f'Create {target_entity} nodes and relationships from {source_entity}',
                    'query': query.strip()
                })
        
        # Update the state
        state['cypher_query_templates'] = templates
        
    except Exception as e:
        logger.error(f"Error generating Cypher templates: {str(e)}")
        state['error_messages'].append(f"Error generating Cypher templates: {str(e)}")
        state['cypher_query_templates'] = []
    
    return state

def synthesize_final_schema_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Final step in schema synthesis: Combine all intermediate results into the final Neo4j schema.
    Uses LLM to synthesize a comprehensive Neo4j schema from all intermediate analysis results.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with inferred_neo4j_schema
    """
    if (state.get('entity_property_consensus') is None or 
        state.get('entity_relationships') is None or
        state.get('entity_properties') is None):
        state['error_messages'].append("Cannot synthesize final schema: missing required data")
        return state
    
    try:
        # Get the necessary data from state
        consensus = state['entity_property_consensus']
        property_entity_mapping = state.get('property_entity_mapping', {})
        entity_relationships = state.get('entity_relationships', [])
        cypher_templates = state.get('cypher_query_templates', [])
        file_name = os.path.basename(state['csv_file_path'])
        primary_entity = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Format the prompt with all intermediate analysis results
        prompt = format_prompt('synthesize_final_schema.txt',
                             file_name=file_name,
                             primary_entity=primary_entity,
                             entity_property_classifications=str(state.get('entity_property_classification', {})),
                             entity_property_consensus=str(consensus),
                             property_entity_mapping=str(property_entity_mapping),
                             entity_relationships=str(entity_relationships),
                             cypher_templates=str(cypher_templates))
        
        # Call the LLM for final schema synthesis
        try:
            response = call_llm_with_json_output(prompt, state_name="synthesize_final_schema")
            
            # Extract the schema results
            schema = {
                "primary_entity_label": response.get('primary_entity_label', primary_entity),
                "columns_classification": response.get('columns_classification', [])
            }
            
            # Add Cypher templates if available in the response
            if 'cypher_templates' in response and isinstance(response['cypher_templates'], list):
                schema['cypher_templates'] = response['cypher_templates']
            elif state.get('cypher_query_templates'):
                schema['cypher_templates'] = state['cypher_query_templates']
            
            # Validate the schema
            if not schema['columns_classification']:
                logger.warning("LLM returned empty columns classification, using fallback")
                raise ValueError("Empty columns classification from LLM")
            
            # Log the schema results
            logger.info(f"Synthesized final schema with {len(schema['columns_classification'])} column classifications")
            
        except Exception as e:
            logger.error(f"Error calling LLM for final schema synthesis: {str(e)}")
            
            # Fallback: Create schema manually based on consensus
            schema = {
                "primary_entity_label": primary_entity,
                "columns_classification": []
            }
            
            # Convert our consensus model to the expected output format
            for column_name, info in consensus.items():
                classification = info['classification']
                
                if classification == 'entity_identifier':
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "PRIMARY_ENTITY_IDENTIFIER",
                        "neo4j_property_key": info['neo4j_property_key'],
                        "uniqueness_ratio": info.get('uniqueness_ratio', 1.0),
                        "semantic_type": info['semantic_type']
                    })
                
                elif classification == 'entity_property':
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "PRIMARY_ENTITY_PROPERTY",
                        "neo4j_property_key": info['neo4j_property_key'],
                        "semantic_type": info['semantic_type']
                    })
                
                elif classification == 'new_entity_type':
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "NEW_NODE_TYPE_VALUES",
                        "new_node_label": info['entity_type'],
                        "neo4j_property_key_for_new_node": "name",
                        "relationship_to_primary": info['relationship_to_primary'],
                        "semantic_type": info['semantic_type']
                    })
                
                elif classification == 'secondary_entity_property':
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "NEW_NODE_PROPERTY",
                        "associated_new_node_label": info['entity_type'],
                        "neo4j_property_key": info['neo4j_property_key'],
                        "semantic_type": info['semantic_type']
                    })
                
                elif classification == 'relationship_property':
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "RELATIONSHIP_PROPERTY",
                        "associated_relationship": info.get('relationship_type', 'UNKNOWN'),
                        "source_node_label": info.get('source_entity', primary_entity),
                        "target_node_label": info.get('target_entity', 'UNKNOWN'),
                        "neo4j_property_key": info['neo4j_property_key'],
                        "semantic_type": info['semantic_type']
                    })
            
            # Add Cypher templates if available
            if state.get('cypher_query_templates'):
                schema['cypher_templates'] = state['cypher_query_templates']
        
        # Update the state with the inferred schema
        state['inferred_neo4j_schema'] = schema
        
    except Exception as e:
        logger.error(f"Error synthesizing final schema: {str(e)}")
        state['error_messages'].append(f"Error synthesizing final schema: {str(e)}")
        # Create a minimal schema as fallback
        state['inferred_neo4j_schema'] = {
            "primary_entity_label": primary_entity,
            "columns_classification": [],
            "error": str(e)
        }
    
    return state

# Helper functions from the original synthesis_nodes.py

def to_neo4j_property_name(column_name: str) -> str:
    """
    Convert a column name to a Neo4j property name (camelCase).
    
    Args:
        column_name: Original column name
        
    Returns:
        Neo4j property name in camelCase
    """
    # Handle empty or None
    if not column_name:
        return "property"
    
    # Split by non-alphanumeric characters
    import re
    words = re.split(r'[^a-zA-Z0-9]', column_name)
    words = [w for w in words if w]  # Remove empty strings
    
    if not words:
        return "property"
    
    # Convert to camelCase
    result = words[0].lower()
    for word in words[1:]:
        if word:
            result += word[0].upper() + word[1:].lower()
    
    return result

def find_associated_entity_type(
    column_name: str,
    column_info: Dict[str, Any],
    all_classifications: Dict[str, Dict[str, Any]],
    entity_types: Set[str],
    all_semantics: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    Find the entity type that this column property should be associated with.
    
    Args:
        column_name: Name of the column being analyzed
        column_info: Classification info for this column
        all_classifications: Classifications of all columns
        entity_types: Set of all entity types
        all_semantics: Semantic analysis of all columns
        
    Returns:
        Associated entity type or None if no association found
    """
    # First, check if the LLM already suggested an association
    column_semantics = all_semantics.get(column_name, {})
    suggested_label = column_semantics.get('new_node_label_suggestion', '')
    
    if suggested_label and suggested_label in entity_types:
        return suggested_label
    
    # Look for naming pattern matches
    for entity_type in entity_types:
        # Skip the primary entity as we're looking for secondary entity associations
        if entity_type == get_primary_entity_from_filename(column_name):
            continue
            
        # Check if column name contains the entity type name
        if entity_type.lower() in column_name.lower():
            return entity_type
        
        # Check for prefix/suffix patterns
        if column_name.lower().startswith(entity_type.lower() + "_"):
            return entity_type
        if column_name.lower().endswith("_" + entity_type.lower()):
            return entity_type
    
    # Look for columns that create entity types
    for other_col, other_info in all_classifications.items():
        if other_info['classification'] == 'new_entity_type':
            entity_type = other_info['entity_type']
            
            # Check if this column seems related to that entity type
            if entity_type.lower() in column_name.lower():
                return entity_type
    
    # No clear association found
    return None

def find_associated_relationship(
    column_name: str,
    column_info: Dict[str, Any],
    all_classifications: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, str]]:
    """
    Find the relationship that this column property should be associated with.
    
    Args:
        column_name: Name of the column being analyzed
        column_info: Classification info for this column
        all_classifications: Classifications of all columns
        
    Returns:
        Dictionary with relationship info or None if no association found
    """
    # Look for columns that define relationships
    for other_col, other_info in all_classifications.items():
        if other_info['classification'] == 'new_entity_type':
            relationship_type = other_info.get('relationship_to_primary', '')
            source_entity = other_info.get('primary_entity', '')
            target_entity = other_info.get('entity_type', '')
            
            # Skip if we don't have complete relationship info
            if not relationship_type or not source_entity or not target_entity:
                continue
            
            # Check if this column seems related to that relationship
            if relationship_type.lower() in column_name.lower():
                return {
                    'relationship': relationship_type,
                    'source_entity': source_entity,
                    'target_entity': target_entity
                }
            
            # Check if column name contains both entity names
            if (source_entity.lower() in column_name.lower() and 
                target_entity.lower() in column_name.lower()):
                return {
                    'relationship': relationship_type,
                    'source_entity': source_entity,
                    'target_entity': target_entity
                }
    
    # No clear association found
    return None
