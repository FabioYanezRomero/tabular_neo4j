"""
Utility functions for the alternative schema synthesis in the Tabular to Neo4j converter.
"""

from typing import Dict, Any, List, Optional
import re
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def to_neo4j_property_name(column_name: str) -> str:
    """
    Convert a column name to a Neo4j property name (camelCase).
    
    Args:
        column_name: Original column name
        
    Returns:
        Neo4j property name in camelCase
    """
    # Handle empty or None input
    if not column_name:
        return "property"
    
    # Convert to string if not already
    column_name = str(column_name)
    
    # Replace non-alphanumeric characters with spaces
    clean_name = re.sub(r'[^a-zA-Z0-9]', ' ', column_name)
    
    # Split by spaces and capitalize each word except the first
    words = clean_name.split()
    if not words:
        return "property"
    
    # First word lowercase, rest capitalized
    camel_case = words[0].lower()
    for word in words[1:]:
        if word:
            camel_case += word[0].upper() + word[1:].lower()
    
    # Ensure it's a valid property name
    if not camel_case[0].isalpha():
        camel_case = "p" + camel_case
    
    return camel_case

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
    logger.debug(f"Finding associated node label for column '{column_name}'")
    
    # Check if the LLM provided a related entity
    related_entity = column_semantics.get('related_entity', '')
    if related_entity and related_entity in new_node_types:
        logger.debug(f"Column '{column_name}' has explicit related entity: '{related_entity}'")
        return related_entity
    
    # Check if the column name contains any node label
    for node_label in new_node_types.keys():
        if node_label.lower() in column_name.lower():
            logger.debug(f"Column '{column_name}' contains node label '{node_label}' in its name")
            return node_label
    
    # Check if the column name is similar to the source column of any node type
    for node_label, source_column in new_node_types.items():
        # Check for name similarity
        if (column_name.lower().startswith(source_column.lower()) or
            source_column.lower().startswith(column_name.lower())):
            logger.debug(f"Column '{column_name}' has name similar to source column '{source_column}' for node '{node_label}'")
            return node_label
    
    logger.debug(f"No associated node label found for column '{column_name}'")
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
    logger.debug(f"Finding associated relationship for column '{column_name}'")
    
    # Look for columns that define relationships (NEW_NODE_TYPE_VALUES)
    for classification in columns_classification:
        if classification.get('role') == 'NEW_NODE_TYPE_VALUES':
            relationship = classification.get('relationship_to_primary', '')
            source_entity = 'PRIMARY'  # Default source is the primary entity
            target_entity = classification.get('new_node_label', '')
            
            # Skip if we don't have complete relationship info
            if not relationship or not target_entity:
                continue
            
            # Check if this column seems related to that relationship
            if relationship.lower() in column_name.lower():
                logger.debug(f"Column '{column_name}' contains relationship '{relationship}' in its name")
                return {
                    'relationship': relationship,
                    'source_entity': source_entity,
                    'target_entity': target_entity
                }
            
            # Check if column name contains the target entity
            if target_entity.lower() in column_name.lower():
                logger.debug(f"Column '{column_name}' contains target entity '{target_entity}' in its name")
                return {
                    'relationship': relationship,
                    'source_entity': source_entity,
                    'target_entity': target_entity
                }
    
    logger.debug(f"No associated relationship found for column '{column_name}'")
    return None
