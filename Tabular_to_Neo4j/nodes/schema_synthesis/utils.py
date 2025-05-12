"""
Utility functions for schema synthesis in the Tabular to Neo4j converter.
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
    logger.debug(f"Finding associated entity type for column '{column_name}'")
    
    # Check if the column name contains any entity type
    for entity_type in entity_types:
        if entity_type.lower() in column_name.lower():
            logger.debug(f"Column '{column_name}' contains entity type '{entity_type}' in its name")
            return entity_type
    
    # Check if any semantic relationships are defined
    column_semantics = all_semantics.get(column_name, {})
    related_entity = column_semantics.get('related_entity', None)
    
    if related_entity:
        # Check if this is a valid entity type
        for entity_type in entity_types:
            if entity_type.lower() == related_entity.lower():
                logger.debug(f"Column '{column_name}' has semantic relationship to entity '{entity_type}'")
                return entity_type
    
    # Check if any entity columns have a similar name pattern
    for other_col, other_info in all_classifications.items():
        if other_col == column_name:
            continue
            
        if other_info.get('classification') in ['entity_identifier', 'new_entity_type']:
            entity_type = other_info.get('entity_type')
            
            # Check for name similarity
            if entity_type and (
                column_name.lower().startswith(entity_type.lower()) or
                entity_type.lower().startswith(column_name.lower()) or
                column_name.lower().endswith(entity_type.lower())
            ):
                logger.debug(f"Column '{column_name}' has name pattern similar to entity '{entity_type}'")
                return entity_type
    
    logger.debug(f"No associated entity type found for column '{column_name}'")
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
    logger.debug(f"Finding associated relationship for column '{column_name}'")
    
    # Look for columns that define relationships
    for other_col, other_info in all_classifications.items():
        if other_col == column_name:
            continue
            
        if other_info.get('classification') == 'new_entity_type':
            relationship_type = other_info.get('relationship_to_primary')
            source_entity = other_info.get('primary_entity')
            target_entity = other_info.get('entity_type')
            
            # Skip if we don't have complete relationship info
            if not relationship_type or not source_entity or not target_entity:
                continue
            
            # Check if this column seems related to that relationship
            if relationship_type.lower() in column_name.lower():
                logger.debug(f"Column '{column_name}' contains relationship type '{relationship_type}' in its name")
                return {
                    'relationship': relationship_type,
                    'source_entity': source_entity,
                    'target_entity': target_entity
                }
            
            # Check if column name contains both entity names
            if (source_entity.lower() in column_name.lower() and 
                target_entity.lower() in column_name.lower()):
                logger.debug(f"Column '{column_name}' contains both source entity '{source_entity}' and target entity '{target_entity}' in its name")
                return {
                    'relationship': relationship_type,
                    'source_entity': source_entity,
                    'target_entity': target_entity
                }
    
    logger.debug(f"No associated relationship found for column '{column_name}'")
    return None
