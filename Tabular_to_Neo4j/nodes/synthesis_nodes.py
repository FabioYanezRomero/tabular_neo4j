"""
Schema synthesis nodes for the LangGraph CSV analysis pipeline.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD

def synthesize_schema_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Synthesize the Neo4j schema based on column analytics and LLM semantic analysis.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with inferred_neo4j_schema
    """
    if (state.get('column_analytics') is None or 
        state.get('llm_column_semantics') is None or 
        state.get('final_header') is None):
        state['error_messages'].append("Cannot synthesize schema: missing required analysis data")
        return state
    
    try:
        # Infer primary entity label from filename
        primary_entity_label = get_primary_entity_from_filename(state['csv_file_path'])
        
        # Initialize the schema structure
        schema = {
            "primary_entity_label": primary_entity_label,
            "columns_classification": []
        }
        
        # Track columns that define new node types for later association
        new_node_types = {}  # {node_label: column_name}
        
        # First pass: Identify primary entity identifiers and new node types
        for column_name in state['final_header']:
            # Get analytics and semantics for this column
            analytics = state.get('column_analytics', {}).get(column_name, {})
            semantics = state.get('llm_column_semantics', {}).get(column_name, {})
            
            # Skip if we don't have both analytics and semantics
            if not analytics or not semantics:
                continue
            
            neo4j_role = semantics.get('neo4j_role', 'UNKNOWN')
            uniqueness_ratio = analytics.get('uniqueness_ratio', 0)
            
            # Rule 1: Identify primary entity identifiers
            if (neo4j_role == 'PRIMARY_ENTITY_IDENTIFIER' and 
                uniqueness_ratio > UNIQUENESS_THRESHOLD):
                schema['columns_classification'].append({
                    "original_column_name": column_name,
                    "role": "PRIMARY_ENTITY_IDENTIFIER",
                    "neo4j_property_key": to_neo4j_property_name(column_name),
                    "uniqueness_ratio": uniqueness_ratio,
                    "semantic_type": semantics.get('semantic_type', 'Unknown')
                })
            
            # Rule 3: Identify columns that should become new node types
            elif neo4j_role == 'NEW_NODE_TYPE_VALUES':
                new_node_label = semantics.get('new_node_label_suggestion', '')
                if not new_node_label:
                    new_node_label = column_name.capitalize()
                
                relationship_type = semantics.get('relationship_type_suggestion', '')
                if not relationship_type:
                    relationship_type = f"HAS_{new_node_label.upper()}"
                
                schema['columns_classification'].append({
                    "original_column_name": column_name,
                    "role": "NEW_NODE_TYPE_VALUES",
                    "new_node_label": new_node_label,
                    "neo4j_property_key_for_new_node": "name",  # Default property name for the new node
                    "relationship_to_primary": relationship_type,
                    "semantic_type": semantics.get('semantic_type', 'Unknown')
                })
                
                # Track this new node type
                new_node_types[new_node_label] = column_name
        
        # Second pass: Process properties and associations
        for column_name in state['final_header']:
            # Skip columns already processed in first pass
            if any(col['original_column_name'] == column_name for col in schema['columns_classification']):
                continue
            
            # Get analytics and semantics for this column
            analytics = state.get('column_analytics', {}).get(column_name, {})
            semantics = state.get('llm_column_semantics', {}).get(column_name, {})
            
            # Skip if we don't have both analytics and semantics
            if not analytics or not semantics:
                continue
            
            neo4j_role = semantics.get('neo4j_role', 'UNKNOWN')
            
            # Rule 2: Process primary entity properties
            if neo4j_role == 'PRIMARY_ENTITY_PROPERTY':
                schema['columns_classification'].append({
                    "original_column_name": column_name,
                    "role": "PRIMARY_ENTITY_PROPERTY",
                    "neo4j_property_key": to_neo4j_property_name(column_name),
                    "semantic_type": semantics.get('semantic_type', 'Unknown')
                })
            
            # Rule 4: Process properties of new node types
            elif neo4j_role == 'NEW_NODE_PROPERTY':
                # Try to find the associated node type
                associated_node_label = find_associated_node_label(
                    column_name, 
                    semantics,
                    new_node_types,
                    state['llm_column_semantics']
                )
                
                if associated_node_label:
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "NEW_NODE_PROPERTY",
                        "associated_new_node_label": associated_node_label,
                        "neo4j_property_key": to_neo4j_property_name(column_name),
                        "semantic_type": semantics.get('semantic_type', 'Unknown')
                    })
                else:
                    # If we can't find an association, treat it as a primary entity property
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "PRIMARY_ENTITY_PROPERTY",
                        "neo4j_property_key": to_neo4j_property_name(column_name),
                        "semantic_type": semantics.get('semantic_type', 'Unknown'),
                        "note": "Originally classified as NEW_NODE_PROPERTY but no association found"
                    })
            
            # Rule 5: Process relationship properties
            elif neo4j_role == 'RELATIONSHIP_PROPERTY':
                # Try to find the associated relationship
                relationship_info = find_associated_relationship(
                    column_name,
                    semantics,
                    schema['columns_classification']
                )
                
                if relationship_info:
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "RELATIONSHIP_PROPERTY",
                        "associated_relationship": relationship_info['relationship'],
                        "source_node_label": primary_entity_label,
                        "target_node_label": relationship_info['target_node_label'],
                        "neo4j_property_key": to_neo4j_property_name(column_name),
                        "semantic_type": semantics.get('semantic_type', 'Unknown')
                    })
                else:
                    # If we can't find an association, treat it as a primary entity property
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "PRIMARY_ENTITY_PROPERTY",
                        "neo4j_property_key": to_neo4j_property_name(column_name),
                        "semantic_type": semantics.get('semantic_type', 'Unknown'),
                        "note": "Originally classified as RELATIONSHIP_PROPERTY but no association found"
                    })
            
            # Handle unknown roles
            elif neo4j_role == 'UNKNOWN':
                # Use heuristics to make a best guess
                data_type = analytics.get('data_type', 'unknown')
                uniqueness_ratio = analytics.get('uniqueness_ratio', 0)
                cardinality = analytics.get('cardinality', 0)
                
                if uniqueness_ratio > UNIQUENESS_THRESHOLD:
                    # Likely an identifier
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "PRIMARY_ENTITY_IDENTIFIER",
                        "neo4j_property_key": to_neo4j_property_name(column_name),
                        "uniqueness_ratio": uniqueness_ratio,
                        "semantic_type": semantics.get('semantic_type', 'Unknown'),
                        "note": "Classified as identifier based on high uniqueness"
                    })
                else:
                    # Default to property
                    schema['columns_classification'].append({
                        "original_column_name": column_name,
                        "role": "PRIMARY_ENTITY_PROPERTY",
                        "neo4j_property_key": to_neo4j_property_name(column_name),
                        "semantic_type": semantics.get('semantic_type', 'Unknown'),
                        "note": "Default classification as property"
                    })
        
        # Update the state with the inferred schema
        state['inferred_neo4j_schema'] = schema
        
    except Exception as e:
        state['error_messages'].append(f"Error synthesizing schema: {str(e)}")
        # Create a minimal schema as fallback
        state['inferred_neo4j_schema'] = {
            "primary_entity_label": get_primary_entity_from_filename(state['csv_file_path']),
            "columns_classification": [],
            "error": str(e)
        }
    
    return state

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
    
    # Split by underscore or space
    parts = column_name.replace('-', '_').replace(' ', '_').split('_')
    
    # Convert to camelCase
    result = parts[0].lower()
    for part in parts[1:]:
        if part:
            result += part[0].upper() + part[1:].lower()
    
    return result

def find_associated_node_label(
    column_name: str,
    column_semantics: Dict[str, Any],
    new_node_types: Dict[str, str],
    all_semantics: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    Find the node label that this column property should be associated with.
    
    Args:
        column_name: Name of the column being analyzed
        column_semantics: Semantic analysis of this column
        new_node_types: Dictionary mapping node labels to their source columns
        all_semantics: Semantic analysis of all columns
        
    Returns:
        Associated node label or None if no association found
    """
    # Check if the column semantics explicitly mentions an association
    if 'new_node_label_suggestion' in column_semantics and column_semantics['new_node_label_suggestion']:
        suggested_label = column_semantics['new_node_label_suggestion']
        if suggested_label in new_node_types:
            return suggested_label
    
    # Look for semantic relationships between this column and node type columns
    for node_label, source_column in new_node_types.items():
        # Check if the column name contains the node label or vice versa
        if (node_label.lower() in column_name.lower() or 
            column_name.lower() in node_label.lower()):
            return node_label
        
        # Check if the column name contains the source column name or vice versa
        if (source_column.lower() in column_name.lower() or 
            column_name.lower() in source_column.lower()):
            return node_label
    
    # If no association found, return None
    return None

def find_associated_relationship(
    column_name: str,
    column_semantics: Dict[str, Any],
    columns_classification: List[Dict[str, Any]]
) -> Optional[Dict[str, str]]:
    """
    Find the relationship that this column property should be associated with.
    
    Args:
        column_name: Name of the column being analyzed
        column_semantics: Semantic analysis of this column
        columns_classification: List of already classified columns
        
    Returns:
        Dictionary with relationship info or None if no association found
    """
    # Check if the column semantics explicitly mentions a relationship
    if 'relationship_type_suggestion' in column_semantics and column_semantics['relationship_type_suggestion']:
        relationship_type = column_semantics['relationship_type_suggestion']
        
        # Find a node type that might be involved in this relationship
        for col in columns_classification:
            if col['role'] == 'NEW_NODE_TYPE_VALUES':
                if 'relationship_to_primary' in col and col['relationship_to_primary'] == relationship_type:
                    return {
                        'relationship': relationship_type,
                        'target_node_label': col['new_node_label']
                    }
        
        # If we have a relationship suggestion but no matching node type,
        # look for any node type to associate with
        for col in columns_classification:
            if col['role'] == 'NEW_NODE_TYPE_VALUES':
                return {
                    'relationship': relationship_type,
                    'target_node_label': col['new_node_label']
                }
    
    # If no association found, return None
    return None
