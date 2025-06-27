"""
Analysis package for the Tabular to Neo4j converter.
This package contains modules for analyzing columns statistically.
"""

from Tabular_to_Neo4j.nodes.intra_table_analysis.column_analytics import perform_column_analytics_node

__all__ = [
    'perform_column_analytics_node'
]
