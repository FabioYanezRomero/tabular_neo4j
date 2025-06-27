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
from Tabular_to_Neo4j.nodes.intra_table_analysis.column_analytics import perform_column_analytics_node
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
        "source": "detect_header",
        "condition": lambda state: (
            "has_header" if state.get("has_header_heuristic", False) else "no_header"
        ),
        "edges": {"has_header": "validate_header", "no_header": "infer_header"},
    },
    ("infer_header", "validate_header"),
    ("validate_header", "detect_header_language"),
    {
        "source": "detect_header_language",
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
        if isinstance(edge, dict):
            graph.add_conditional_edges(edge["source"], edge["condition"], edge["edges"])
        else:
            graph.add_edge(*edge)
    graph.set_entry_point(ENTRY_POINT)
    graph.set_finish_point(END)
    return graph


def run_pipeline(graph, input_path):
    import os
    import time
    from Tabular_to_Neo4j.app_state import GraphState
    from Tabular_to_Neo4j.utils.logging_config import get_logger
    logger = get_logger(__name__)
    if not os.path.exists(input_path):
        logger.error(f"Input path not found: {input_path}")
        raise FileNotFoundError(f"Input path not found: {input_path}")
    file_size = os.path.getsize(input_path) / 1024  # KB
    logger.info(f"File size: {file_size:.2f} KB")
    # Create and run the graph
    logger.debug("Creating state graph")
    app = graph.compile()
    logger.debug("Initializing state")
    initial_state = GraphState(csv_file_path=input_path, error_messages=[])
    logger.info(f"Executing analysis pipeline")
    start_time = time.time()
    try:
        final_state = app.invoke(initial_state)
        execution_time = time.time() - start_time
        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise
    return final_state
