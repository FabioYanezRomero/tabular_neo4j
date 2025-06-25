"""
Graph definition for single-table workflows in Tabular_to_Neo4j.
"""
from langgraph.graph import StateGraph, END
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.nodes.input import load_csv_node, detect_header_heuristic_node
from Tabular_to_Neo4j.nodes.header_processing import (
    infer_header_llm_node,
    validate_header_llm_node,
    detect_header_language_node,
    translate_header_llm_node,
    apply_header_node,
)
from Tabular_to_Neo4j.nodes.intra-table-analysis.column_analytics import perform_column_analytics_node
from Tabular_to_Neo4j.nodes.entity_inference import (
    classify_entities_properties_node,
    reconcile_entity_property_node,
    map_properties_to_entities_node,
    infer_entity_relationships_node,
)
from Tabular_to_Neo4j.nodes.db_schema import generate_cypher_templates_node

# Ordered list of nodes in the pipeline. The position in this list defines the order used for output file naming.
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
    ("generate_cypher_templates", generate_cypher_templates_node),
]

# Declarative edge definitions. A tuple represents a direct edge.
# A dictionary with 'condition' defines conditional edges using a callable that returns a key mapping to the next node.
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
    ("infer_entity_relationships", "generate_cypher_templates"),
    ("generate_cypher_templates", END),
]

# Entry point of the graph (first node in PIPELINE_NODES)
ENTRY_POINT = PIPELINE_NODES[0][0]

def create_single_table_graph() -> StateGraph:
    """
    Create the LangGraph for a single-table workflow.
    Returns:
        StateGraph instance
    """
    graph = StateGraph(GraphState)
    for node_name, node_func in PIPELINE_NODES:
        graph.add_node(node_name, node_func)
    for edge in PIPELINE_EDGES:
        graph.add_edge(**edge) if isinstance(edge, dict) else graph.add_edge(*edge)
    graph.set_entry_point(ENTRY_POINT)
    graph.set_finish_point(END)
    return graph
