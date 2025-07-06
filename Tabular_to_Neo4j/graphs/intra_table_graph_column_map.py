"""Graph for per-table analysis ending with per-column graph-element mapping.
This focuses on intra-table steps only (no cross-table logic).
"""
from langgraph.graph import StateGraph, END
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.nodes.input import load_csv_node


from Tabular_to_Neo4j.nodes.Intra_table_nodes import (
    detect_table_entities_node,
    infer_intra_table_relations_node,
    map_column_to_graph_element_node,
)

# Ordered list defines node order for output naming
PIPELINE_NODES = [
    ("load_csv", load_csv_node),

    ("detect_table_entities", detect_table_entities_node),
    ("infer_intra_table_relations", infer_intra_table_relations_node),
    ("map_column_to_graph_element", map_column_to_graph_element_node),
]

PIPELINE_EDGES = [
    ("load_csv", "detect_table_entities"),
    ("detect_table_entities", "infer_intra_table_relations"),
    ("infer_intra_table_relations", "map_column_to_graph_element"),
    ("map_column_to_graph_element", END),
]

ENTRY_POINT = PIPELINE_NODES[0][0]


def create_intra_table_column_map_graph() -> StateGraph:
    import logging, traceback
    logger = logging.getLogger(__name__)
    try:
        graph = StateGraph(GraphState)
        for name, func in PIPELINE_NODES:
            graph.add_node(name, func)
        for edge in PIPELINE_EDGES:
            if isinstance(edge, dict):
                graph.add_conditional_edges(edge["source"], edge["condition"], edge["edges"])
            else:
                graph.add_edge(*edge)
        graph.set_entry_point(ENTRY_POINT)
        graph.set_finish_point("map_column_to_graph_element")

        # For output ordering
        from Tabular_to_Neo4j.utils.output_saver import output_saver
        output_saver.set_node_order_map(PIPELINE_NODES)
        return graph
    except Exception as e:
        logger.error("Graph construction error: %s", e)
        logger.error(traceback.format_exc())
        raise
