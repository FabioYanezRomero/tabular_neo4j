"""
Input package for the Tabular to Neo4j converter.
This package contains modules for loading and initial processing of CSV files.
"""

from Tabular_to_Neo4j.nodes.input.csv_loader import load_csv_node
from Tabular_to_Neo4j.nodes.input.header_detection import detect_header_heuristic_node

__all__ = [
    'load_csv_node',
    'detect_header_heuristic_node'
]
