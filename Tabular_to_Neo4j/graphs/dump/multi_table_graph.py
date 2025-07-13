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
from Tabular_to_Neo4j.nodes.cross_table_analysis.llm_relation_node import llm_relation_node
import os
from typing import Dict, Any, Optional
from Tabular_to_Neo4j.utils.output_saver import output_saver
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
    ("map_properties_to_entity", map_properties_to_entities_node),
    ("infer_entity_relationships", infer_entity_relationships_node),
]

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
    ("reconcile_entity_property", "map_properties_to_entity"),
    ("map_properties_to_entity", "infer_entity_relationships"),
]

# Cross-table nodes to be run after all per-table nodes have finished
CROSS_TABLE_NODES = [
    ("columns_contextualization", columns_contextualization_node),
    ("semantic_embedding", semantic_embedding_node),
    ("llm_relation", llm_relation_node),
]

ENTRY_POINT = PIPELINE_NODES[0][0]

def create_multi_table_graph() -> StateGraph:
    """
    Returns the pipeline structure (StateGraph) for a single table in the multi-table workflow.
    This is analogous to create_single_table_graph, but for use in multi-table contexts.
    The folder path and table enumeration are handled elsewhere (initialize_multi_table_state and pipeline runner).
    """
    graph = StateGraph(GraphState)

    # LangGraph expects each node callable to accept (state: GraphState) (and optionally a
    # RunnableConfig/StreamWriter). Our node functions also need the `node_order` so they
    # can produce deterministic file names and logs.  We therefore create a small wrapper
    # around every node function that injects the `node_order`, while still presenting the
    # correct callable signature to LangGraph's type checker and runtime.
    from functools import partial
    from langchain_core.runnables import RunnableConfig  # runtime import, avoids hard dep here

    for idx, (node_name, node_func) in enumerate(PIPELINE_NODES, start=1):
        # Each wrapper only takes the state (plus optional config) and forwards the call
        # to the original implementation with the captured node index.
        def _make_wrapper(func, order):
            def _wrapper(state: GraphState, config: RunnableConfig | None = None):  # type: ignore[override]
                # Pass through to real implementation with `node_order`.
                return func(state, node_order=order)
            return _wrapper

        graph.add_node(node_name, _make_wrapper(node_func, idx))
    for edge in PIPELINE_EDGES:
        if isinstance(edge, dict):
            graph.add_conditional_edges(edge["source"], edge["condition"], edge["edges"])
        else:
            graph.add_edge(*edge)
    graph.set_entry_point(ENTRY_POINT)
    graph.set_finish_point("infer_entity_relationships")

    # Configure output_saver with node order mapping so downstream nodes can
    # create deterministic file names using the same ordering we defined above.
    output_saver.set_node_order_map(PIPELINE_NODES)
    return graph
        

def run_multi_table_pipeline(table_folder: str, config: Optional[Dict[str, Any]] = None) -> MultiTableGraphState:
    import logging
    logger = logging.getLogger(__name__)

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
    
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized. All output saving must use the same timestamp for the run.")

    # Per-table phase (use the StateGraph abstraction for each table)
    state = initialize_multi_table_state(table_folder)

    # For each table, run the compiled graph node-by-node, saving state and prompts after each node
    for table_name, table_state in state.items():
        logger.info(f'[PER_TABLE][{table_name}][BEFORE_PIPELINE] Initial state type: {type(table_state).__name__}')
        # Manually step through nodes
        current_state = table_state
        for node_idx, (node_name, node_func) in enumerate(PIPELINE_NODES, 1):
            try:
                current_state = node_func(current_state, node_order=node_idx)
            except Exception as e:
                logger.error(f'[PER_TABLE][{table_name}][NODE_EXEC][{node_name}] Exception: {e}, State type: {type(current_state).__name__}, State value: {current_state}')
                raise

        # Update state for this table
        state[table_name] = current_state
        logger.info(f'[PER_TABLE][{table_name}][AFTER_PIPELINE] State type: {type(current_state).__name__}')

    # Diagnostic logging: print types after per-table pipelines, before cross-table nodes
    logger.info('[PER_TABLE][AFTER] Table states: ' + str({k: type(v).__name__ for k, v in state.items()}))
    # Enforce that all table states are valid mapping types (GraphState, AddableValuesDict, or MutableMapping)
    from collections.abc import MutableMapping
    try:
        from langchain_core.utils import AddableValuesDict  # type: ignore
    except ImportError:  # pragma: no cover
        AddableValuesDict = None  # type: ignore

    # Build a tuple of allowed types for isinstance that passes static type checking
    per_table_allowed_types: tuple[type, ...] = (GraphState, MutableMapping)
    if AddableValuesDict is not None and isinstance(AddableValuesDict, type):  # runtime safety check
        per_table_allowed_types = (*per_table_allowed_types, AddableValuesDict)  # type: ignore[arg-type]

    for table_name, table_state in state.items():
        if not isinstance(table_state, per_table_allowed_types):
            logger.error(
                f"[PER_TABLE][{table_name}] Invalid state type after pipeline: {type(table_state).__name__}"
            )
            raise TypeError(
                f"Table state for '{table_name}' must be one of {[t.__name__ for t in per_table_allowed_types]}, got {type(table_state)}"
            )
    
    # Cross-table phase
    for node_idx, (node_name, node_func) in enumerate(CROSS_TABLE_NODES, 1):
        logger.info(f'[CROSS_TABLE][{node_name}][BEFORE] Table states: ' + str({k: type(v).__name__ for k, v in state.items()}))
        real_idx = node_idx + len(PIPELINE_NODES)
        state = node_func(state, node_order=real_idx)
        # Save cross-table node output
        if output_saver:
            output_saver.save_node_output(node_name, state, node_order=node_idx, table_name="inter_table")
        # Log type and truncated value for each table after cross-table node
        for k, v in state.items():
            short_val = str(v)
            if len(short_val) > 300:
                short_val = short_val[:300] + '...'
            logger.info(f'[CROSS_TABLE][{node_name}][NODE_EXEC][AFTER][{k}] Type: {type(v).__name__}, Value: {short_val}')
        logger.info(f'[CROSS_TABLE][{node_name}][AFTER] Table states: ' + str({k: type(v).__name__ for k, v in state.items()}))
    # Ensure all table states are valid mapping types (GraphState, AddableValuesDict, or MutableMapping) after cross-table nodes
    from collections.abc import MutableMapping
    try:
        from langchain_core.utils import AddableValuesDict  # type: ignore
    except ImportError:  # pragma: no cover
        AddableValuesDict = None  # type: ignore

    allowed_types_final: tuple[type, ...] = (GraphState, MutableMapping)
    if AddableValuesDict is not None and isinstance(AddableValuesDict, type):  # runtime safety check
        allowed_types_final = (*allowed_types_final, AddableValuesDict)  # type: ignore[arg-type]

    for table_name, table_state in state.items():
        if not isinstance(table_state, allowed_types_final):
            logger.error(
                f"[FINAL][{table_name}] Invalid state type after cross-table nodes: {type(table_state).__name__}"
            )
            raise TypeError(
                f"Table state for '{table_name}' must be one of {[t.__name__ for t in allowed_types_final]} after cross-table nodes, got {type(table_state)}"
            )
    return state
