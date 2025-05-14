"""
Utility functions for entity inference in the Tabular to Neo4j converter.
"""

from typing import Dict, Any
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

# This file now only contains the to_neo4j_property_name function
# Other helper functions have been removed as part of the simplification
