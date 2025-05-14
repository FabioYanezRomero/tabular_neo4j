"""
Entity Inference package for the Tabular to Neo4j converter.
This package contains modules for identifying entities, properties, and their relationships.
"""

from Tabular_to_Neo4j.nodes.entity_inference.entity_classification import classify_entities_properties_node
from Tabular_to_Neo4j.nodes.entity_inference.entity_reconciliation import reconcile_entity_property_node
from Tabular_to_Neo4j.nodes.entity_inference.property_mapping import map_properties_to_entities_node
from Tabular_to_Neo4j.nodes.entity_inference.relationship_inference import infer_entity_relationships_node
from Tabular_to_Neo4j.nodes.entity_inference.utils import (
    to_neo4j_property_name,
    find_associated_entity_type,
    find_associated_relationship
)

__all__ = [
    'classify_entities_properties_node',
    'reconcile_entity_property_node',
    'map_properties_to_entities_node',
    'infer_entity_relationships_node',
    'to_neo4j_property_name',
    'find_associated_entity_type',
    'find_associated_relationship'
]
