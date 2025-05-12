"""
Main module for schema synthesis in the Tabular to Neo4j converter.
This module provides a simplified interface to the schema synthesis process,
importing and re-exporting all the schema synthesis nodes from their respective modules.
"""

# Import all schema synthesis nodes from their respective modules
from Tabular_to_Neo4j.nodes.schema_synthesis import (
    classify_entities_properties_node,
    reconcile_entity_property_node,
    map_properties_to_entities_node,
    infer_entity_relationships_node,
    generate_cypher_templates_node,
    synthesize_final_schema_node
)

# Re-export all nodes for easy import
__all__ = [
    'classify_entities_properties_node',
    'reconcile_entity_property_node',
    'map_properties_to_entities_node',
    'infer_entity_relationships_node',
    'generate_cypher_templates_node',
    'synthesize_final_schema_node'
]
