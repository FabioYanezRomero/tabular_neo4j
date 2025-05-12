"""
Alternative schema synthesis package for the Tabular to Neo4j converter.
This package contains an alternative implementation of schema synthesis.
"""

from Tabular_to_Neo4j.nodes.alternative_synthesis.schema_synthesis import synthesize_schema_node
from Tabular_to_Neo4j.nodes.alternative_synthesis.utils import (
    to_neo4j_property_name,
    find_associated_node_label,
    find_associated_relationship
)

__all__ = [
    'synthesize_schema_node',
    'to_neo4j_property_name',
    'find_associated_node_label',
    'find_associated_relationship'
]
