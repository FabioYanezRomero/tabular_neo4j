"""
Graph definition for multi-table workflows in Tabular_to_Neo4j.
Each table's state is separated in a dictionary keyed by table name.
This graph processes all tables in a folder up to the `infer_entity_relationships` node.
"""
from langgraph.graph import StateGraph, END
from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.nodes.input import load_csv_node, detect_header_heuristic_node
from Tabular_to_Neo4j.nodes.header_processing import (
    infer_header_llm_node,
    validate_header_llm_node,
    detect_header_language_node,
    translate_header_llm_node,
    apply_header_node,
)
from Tabular_to_Neo4j.nodes.intra_table_analysis import perform_column_analytics_node
from Tabular_to_Neo4j.nodes.entity_inference import (
    classify_entities_properties_node,
    reconcile_entity_property_node,
    map_properties_to_entities_node,
    infer_entity_relationships_node,
)
from Tabular_to_Neo4j.nodes.cross_table_analysis.columns_contextualization import columns_contextualization_node
from Tabular_to_Neo4j.nodes.cross_table_analysis.semantic_embedding_node import semantic_embedding_node
import os
from typing import Dict, Any
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_path_for_csv

# Only up to infer_entity_relationships
PIPELINE_NODES = [
    ("load_csv", load_csv_node),
    ("detect_header", detect_header_heuristic_node),
    ("infer_header", infer_header_llm_node),
    ("validate_header", validate_header_llm_node),
    ("detect_header_language", detect_header_language_node),
    ("translate_header", translate_header_llm_node),
    ("apply_header", apply_header_node),
    ("analyze_columns", perform_column_analytics_node),
    ("classify_entities_properties", classify_entities_properties_node),
    ("reconcile_entity_property", reconcile_entity_property_node),
    ("map_properties_to_entities", map_properties_to_entities_node),
    ("infer_entity_relationships", infer_entity_relationships_node),
]

PIPELINE_EDGES = [
    ("load_csv", "detect_header"),
    {
        "from": "detect_header",
        "condition": lambda state: (
            "has_header" if state.get("has_header_heuristic", False) else "no_header"
        ),
        "edges": {"has_header": "validate_header", "no_header": "infer_header"},
    },
    ("infer_header", "validate_header"),
    ("validate_header", "detect_header_language"),
    {
        "from": "detect_header_language",
        "condition": lambda state: (
            "same_language" if state.get("is_header_in_target_language", False) else "different_language"
        ),
        "edges": {"same_language": "apply_header", "different_language": "translate_header"},
    },
    ("translate_header", "apply_header"),
    ("apply_header", "analyze_columns"),
    ("analyze_columns", "classify_entities_properties"),
    ("classify_entities_properties", "reconcile_entity_property"),
    ("reconcile_entity_property", "map_properties_to_entities"),
    ("map_properties_to_entities", "infer_entity_relationships"),
]

ENTRY_POINT = PIPELINE_NODES[0][0]



def create_multi_table_graph(table_folder: str) -> Dict[str, StateGraph]:
    """
    For each CSV file in the given folder, create a graph up to infer_entity_relationships node.
    Returns a dict mapping table name to its StateGraph.
    After all per-table graphs are run, cross-table nodes (such as columns_contextualization) should be run on the MultiTableGraphState.
    """
    graphs = {}
    for fname in os.listdir(table_folder):
        if fname.lower().endswith('.csv'):
            table_name = os.path.splitext(fname)[0]
            graph = StateGraph(GraphState)
            for node_name, node_func in PIPELINE_NODES:
                graph.add_node(node_name, node_func)
            for edge in PIPELINE_EDGES:
                graph.add_edge(**edge) if isinstance(edge, dict) else graph.add_edge(*edge)
            graph.set_entry_point(ENTRY_POINT)
            graph.set_finish_point("infer_entity_relationships")
            graphs[table_name] = graph
    return graphs


# Cross-table nodes to be run after all per-table nodes have finished
CROSS_TABLE_NODES = [
    ("columns_contextualization", columns_contextualization_node),
    ("semantic_embedding", semantic_embedding_node),
]

def run_multi_table_pipeline(state: MultiTableGraphState, config: Dict[str, Any] = None) -> MultiTableGraphState:
    """
    Runs the full multi-table pipeline:
    1. For each table, runs all per-table nodes in sequence.
    2. After all tables are processed, runs all cross-table nodes.
    Returns the final MultiTableGraphState.
    """
    # Per-table phase
    for table_name, table_state in state.items():
        for node_name, node_func in PIPELINE_NODES:
            table_state = node_func(table_state, config)
            state[table_name] = table_state
    # Cross-table phase
    for node_name, node_func in CROSS_TABLE_NODES:
        state = node_func(state, config)
    return state

def initialize_multi_table_state(table_folder: str) -> MultiTableGraphState:
    """
    Initializes a MultiTableGraphState for all tables in the folder.
    For each CSV, also attempts to find and store the corresponding metadata file path (as in the one-table case).
    Each key is a table name, and the value is a full GraphState instance.
    """
    state = MultiTableGraphState()
    for fname in os.listdir(table_folder):
        if fname.lower().endswith('.csv'):
            table_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(table_folder, fname)
            metadata_path = get_metadata_path_for_csv(csv_path)
            state[table_name] = GraphState(csv_file_path=csv_path, metadata_file_path=metadata_path)
    return state
