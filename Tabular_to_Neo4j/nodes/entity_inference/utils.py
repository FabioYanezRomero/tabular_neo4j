"""
Utility functions for entity inference in the Tabular to Neo4j converter.
"""

from typing import Dict, Any, List, Optional, Set
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

def find_associated_entity_type(
    column_name: str,
    column_info: Dict[str, Any],
    all_classifications: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Find the entity type that this column property should be associated with.
    
    Args:
        column_name: Name of the column being analyzed
        column_info: Classification info for this column
        all_classifications: Classifications of all columns
        
    Returns:
        Dictionary with entity info or None if no association found
    """
    # If this is an entity itself, return None (no association needed)
    if column_info.get('type') == 'entity':
        return None
    
    # If this column has an explicit entity association, use that
    if 'associated_entity' in column_info and column_info['associated_entity']:
        entity_name = column_info['associated_entity']
        
        # Find the entity in all classifications
        for col, info in all_classifications.items():
            if (info.get('type') == 'entity' and 
                info.get('entity_type', '').lower() == entity_name.lower()):
                return {
                    'column_name': col,
                    'entity_type': info.get('entity_type')
                }
    
    # Default: associate with the primary entity
    for col, info in all_classifications.items():
        if info.get('type') == 'entity' and info.get('is_primary', False):
            return {
                'column_name': col,
                'entity_type': info.get('entity_type')
            }
    
    # If no primary entity found, use the first entity
    for col, info in all_classifications.items():
        if info.get('type') == 'entity':
            return {
                'column_name': col,
                'entity_type': info.get('entity_type')
            }
    
    # No entity found
    return None

def find_associated_relationship(
    column_name: str,
    column_info: Dict[str, Any],
    all_classifications: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Find the relationship that this column property should be associated with.
    
    Args:
        column_name: Name of the column being analyzed
        column_info: Classification info for this column
        all_classifications: Classifications of all columns
        
    Returns:
        Dictionary with relationship info or None if no association found
    """
    # If this is an entity or has no relationship association, return None
    if (column_info.get('type') == 'entity' or 
        'associated_relationship' not in column_info or 
        not column_info['associated_relationship']):
        return None
    
    # Get the relationship name
    relationship_name = column_info['associated_relationship']
    
    # Find the source and target entities for this relationship
    source_entity = None
    target_entity = None
    
    # First, find the primary entity as a default source
    for col, info in all_classifications.items():
        if info.get('type') == 'entity' and info.get('is_primary', False):
            source_entity = {
                'column_name': col,
                'entity_type': info.get('entity_type')
            }
            break
    
    # Then look for the target entity based on the relationship name
    for col, info in all_classifications.items():
        if (info.get('type') == 'entity' and 
            relationship_name.lower().endswith(info.get('entity_type', '').lower())):
            target_entity = {
                'column_name': col,
                'entity_type': info.get('entity_type')
            }
            break
    
    # If we found both source and target, return the relationship info
    if source_entity and target_entity:
        return {
            'source_entity': source_entity['entity_type'],
            'target_entity': target_entity['entity_type'],
            'relationship_type': relationship_name
        }
    
    # No relationship found
    return None
