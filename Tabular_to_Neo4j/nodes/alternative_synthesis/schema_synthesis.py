"""
Alternative schema synthesis module for the Tabular to Neo4j converter.
This module provides an alternative implementation of schema synthesis.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.nodes.alternative_synthesis.utils import (
    to_neo4j_property_name,
    find_associated_node_label,
    find_associated_relationship
)

# Configure logging
logger = get_logger(__name__)

def synthesize_schema_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Synthesize the Neo4j schema based on column analytics and LLM semantic analysis.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with inferred_neo4j_schema
    """
    logger.info("Starting alternative schema synthesis process")
    
    if (state.get('column_analytics') is None or 
        state.get('llm_column_semantics') is None or 
        state.get('final_header') is None):
        error_msg = "Cannot synthesize schema: missing required analysis data"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    try:
        # Infer primary entity label from filename
        primary_entity_label = get_primary_entity_from_filename(state['csv_file_path'])
        logger.info(f"Primary entity label: {primary_entity_label}")
        
        # Get analytics and semantics
        analytics = state['column_analytics']
        semantics = state['llm_column_semantics']
        
        # Initialize schema components
        columns_classification = []
        new_node_types = {}  # Maps node labels to their source columns
        relationship_properties = {}  # Maps relationship types to their properties
        
        # First pass: classify columns based on analytics and semantics
        logger.info("Classifying columns based on analytics and semantics")
        for column_name, column_analytics in analytics.items():
            column_semantics = semantics.get(column_name, {})
            
            # Default classification
            classification = {
                'original_column_name': column_name,
                'neo4j_property_key': to_neo4j_property_name(column_name),
                'role': 'PRIMARY_ENTITY_PROPERTY',
                'data_type': column_analytics.get('data_type', 'STRING'),
                'semantic_type': column_semantics.get('semantic_type', 'Unknown'),
                'description': column_semantics.get('description', ''),
            }
            
            # Check if this column is unique enough to be an identifier
            uniqueness = column_analytics.get('uniqueness', 0)
            if uniqueness > UNIQUENESS_THRESHOLD:
                classification['role'] = 'PRIMARY_ENTITY_IDENTIFIER'
                logger.debug(f"Column '{column_name}' classified as PRIMARY_ENTITY_IDENTIFIER (uniqueness: {uniqueness:.2f})")
            
            # Check if LLM suggests this is a new node type
            neo4j_role = column_semantics.get('neo4j_role', '')
            if neo4j_role == 'NEW_NODE_TYPE':
                # This column contains values that should be new nodes
                related_entity = column_semantics.get('related_entity', '')
                if related_entity:
                    new_node_label = related_entity
                else:
                    # Derive node label from column name if not provided
                    new_node_label = ''.join(word.capitalize() for word in column_name.split('_'))
                
                classification['role'] = 'NEW_NODE_TYPE_VALUES'
                classification['new_node_label'] = new_node_label
                classification['relationship_to_primary'] = f"HAS_{new_node_label.upper()}"
                
                # Register this new node type
                new_node_types[new_node_label] = column_name
                logger.debug(f"Column '{column_name}' classified as NEW_NODE_TYPE_VALUES with label '{new_node_label}'")
            
            # Check if LLM suggests this is a property of a secondary node
            elif neo4j_role == 'SECONDARY_NODE_PROPERTY':
                # Try to find which node this property belongs to
                associated_node = find_associated_node_label(
                    column_name, 
                    column_semantics,
                    new_node_types,
                    semantics
                )
                
                if associated_node:
                    classification['role'] = 'NEW_NODE_PROPERTY'
                    classification['associated_new_node_label'] = associated_node
                    logger.debug(f"Column '{column_name}' classified as NEW_NODE_PROPERTY for '{associated_node}'")
                else:
                    # Default to primary entity property if no association found
                    classification['role'] = 'PRIMARY_ENTITY_PROPERTY'
                    classification['note'] = "Could not determine associated node type"
                    logger.warning(f"Could not determine associated node type for '{column_name}', defaulting to PRIMARY_ENTITY_PROPERTY")
            
            # Check if LLM suggests this is a relationship property
            elif neo4j_role == 'RELATIONSHIP_PROPERTY':
                # Try to find which relationship this property belongs to
                relationship_info = find_associated_relationship(
                    column_name,
                    column_semantics,
                    columns_classification
                )
                
                if relationship_info:
                    classification['role'] = 'RELATIONSHIP_PROPERTY'
                    classification['relationship_type'] = relationship_info['relationship']
                    classification['source_node_label'] = relationship_info['source_entity']
                    classification['target_node_label'] = relationship_info['target_entity']
                    
                    # Register this property with the relationship
                    rel_key = f"{relationship_info['source_entity']}_{relationship_info['relationship']}_{relationship_info['target_entity']}"
                    if rel_key not in relationship_properties:
                        relationship_properties[rel_key] = []
                    relationship_properties[rel_key].append(column_name)
                    
                    logger.debug(f"Column '{column_name}' classified as RELATIONSHIP_PROPERTY for '{rel_key}'")
                else:
                    # Default to primary entity property if no association found
                    classification['role'] = 'PRIMARY_ENTITY_PROPERTY'
                    classification['note'] = "Could not determine associated relationship"
                    logger.warning(f"Could not determine associated relationship for '{column_name}', defaulting to PRIMARY_ENTITY_PROPERTY")
            
            # Add to our classifications
            columns_classification.append(classification)
        
        # Build the Neo4j schema
        logger.info("Building Neo4j schema from column classifications")
        schema = {
            'primary_entity_label': primary_entity_label,
            'columns_classification': columns_classification,
            'new_node_types': new_node_types,
            'relationship_properties': relationship_properties,
            'cypher_templates': generate_cypher_templates(
                primary_entity_label,
                columns_classification,
                state['csv_file_path']
            )
        }
        
        # Update the state with the inferred schema
        state['inferred_neo4j_schema'] = schema
        logger.info("Successfully synthesized Neo4j schema")
        
    except Exception as e:
        error_msg = f"Error synthesizing Neo4j schema: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    return state

def generate_cypher_templates(
    primary_entity_label: str,
    columns_classification: List[Dict[str, Any]],
    csv_file_path: str
) -> Dict[str, Any]:
    """
    Generate Cypher query templates for the inferred schema.
    
    Args:
        primary_entity_label: The label for the primary entity
        columns_classification: List of column classifications
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary of Cypher query templates
    """
    logger.info("Generating Cypher query templates")
    
    # Extract file name from path
    import os
    file_name = os.path.basename(csv_file_path)
    
    # Group columns by role
    columns_by_role = {}
    for col in columns_classification:
        role = col.get('role', 'UNKNOWN')
        if role not in columns_by_role:
            columns_by_role[role] = []
        columns_by_role[role].append(col)
    
    # Generate LOAD CSV query
    load_query_parts = [
        f"LOAD CSV WITH HEADERS FROM 'file:///{file_name}' AS row"
    ]
    
    # Create primary entity nodes
    primary_entity_props = []
    
    # Add identifier properties
    if 'PRIMARY_ENTITY_IDENTIFIER' in columns_by_role:
        for col in columns_by_role['PRIMARY_ENTITY_IDENTIFIER']:
            primary_entity_props.append(
                f"{col['neo4j_property_key']}: row.`{col['original_column_name']}`"
            )
    
    # Add regular properties
    if 'PRIMARY_ENTITY_PROPERTY' in columns_by_role:
        for col in columns_by_role['PRIMARY_ENTITY_PROPERTY']:
            primary_entity_props.append(
                f"{col['neo4j_property_key']}: row.`{col['original_column_name']}`"
            )
    
    # Create the primary entity
    load_query_parts.append(
        f"CREATE (e:{primary_entity_label} {{{', '.join(primary_entity_props)}}})"
    )
    
    # Create new node types and relationships
    if 'NEW_NODE_TYPE_VALUES' in columns_by_role:
        for col in columns_by_role['NEW_NODE_TYPE_VALUES']:
            node_label = col['new_node_label']
            rel_type = col['relationship_to_primary']
            
            # Find properties for this node type
            node_props = []
            if 'NEW_NODE_PROPERTY' in columns_by_role:
                for prop_col in columns_by_role['NEW_NODE_PROPERTY']:
                    if prop_col.get('associated_new_node_label') == node_label:
                        node_props.append(
                            f"{prop_col['neo4j_property_key']}: row.`{prop_col['original_column_name']}`"
                        )
            
            # Add the value itself as a property
            node_props.append(f"value: row.`{col['original_column_name']}`")
            
            # Create the node and relationship
            load_query_parts.append(
                f"CREATE (n:{node_label} {{{', '.join(node_props)}}})"
            )
            load_query_parts.append(
                f"CREATE (e)-[:{rel_type}]->(n)"
            )
    
    # Combine all parts into the final query
    load_query = "\n".join(load_query_parts)
    
    # Generate constraint queries
    constraint_queries = []
    
    # Add uniqueness constraints for identifiers
    if 'PRIMARY_ENTITY_IDENTIFIER' in columns_by_role:
        for col in columns_by_role['PRIMARY_ENTITY_IDENTIFIER']:
            constraint_queries.append(
                f"CREATE CONSTRAINT {primary_entity_label}_{col['neo4j_property_key']}_unique IF NOT EXISTS\n"
                f"FOR (n:{primary_entity_label})\n"
                f"REQUIRE n.{col['neo4j_property_key']} IS UNIQUE"
            )
    
    # Generate example queries
    example_queries = [
        f"MATCH (n:{primary_entity_label}) RETURN n LIMIT 10",
        f"MATCH (n:{primary_entity_label})-[r]->(m) RETURN n, r, m LIMIT 10"
    ]
    
    return {
        'load_query': load_query,
        'constraint_queries': constraint_queries,
        'example_queries': example_queries
    }
