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

use_analytics = False

# Ordered list defines node order for output naming
PIPELINE_NODES = [
    ("load_csv", load_csv_node),

    ("detect_table_entities", detect_table_entities_node),
    ("infer_intra_table_relations", infer_intra_table_relations_node),
    ("map_column_to_graph_element", map_column_to_graph_element_node),
]

from typing import Union, Tuple, Dict, Any

# Heterogeneous collection: either simple (source, target) tuple or a conditional-edge dict
PipeEdge = Union[Tuple[str, str], Dict[str, Any]]

PIPELINE_EDGES: list[PipeEdge] = [
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
        # LangGraph `StateGraph` expects each node callable to take only a `GraphState` (and
        # optionally a `RunnableConfig`). Our underlying node functions also take a
        # `node_order` argument which LangGraph does **not** pass. We therefore wrap each
        # node so that the public signature conforms to LangGraph while still forwarding
        # the correct `node_order` to the real implementation.
        for idx, (name, func) in enumerate(PIPELINE_NODES, 1):
            def _make_node(f, order):
                def _node(state: GraphState):  # type: ignore[override]
                    return f(state, node_order=order, use_analytics=use_analytics)
                return _node
            graph.add_node(name, _make_node(func, idx))
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

# ---------------- Multi-table runner combining intra- and inter-table logic -----------------
from typing import Dict, Any, Optional
import os
from Tabular_to_Neo4j.app_state import MultiTableGraphState
from Tabular_to_Neo4j.nodes.inter_table_nodes import (
    merge_synonym_entities_node,
    merge_relation_types_node,
)
from Tabular_to_Neo4j.utils.output_saver import output_saver
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_path_for_csv

def run_column_map_multi_table_pipeline(table_folder: str, config: Optional[Dict[str, Any]] = None, use_analytics: bool = True) -> MultiTableGraphState:
    """Runs the column-mapping intra-table graph for every CSV in *table_folder*, then executes
    all inter-table nodes in sequence and returns the resulting MultiTableGraphState.
    Output is saved via the global `output_saver` after each node for traceability.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not output_saver:
        raise RuntimeError("OutputSaver not initialised – call init_output_saver() first")

    # --- initialise state per table
    state: MultiTableGraphState = MultiTableGraphState()
    for fname in os.listdir(table_folder):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(table_folder, fname)
            meta_path = get_metadata_path_for_csv(csv_path)
            state[table_name] = GraphState(csv_file_path=csv_path, metadata_file_path=meta_path)

    # --- per-table phase
    intra_nodes = PIPELINE_NODES  # from this module
    for table_name, tbl_state in state.items():
        current = tbl_state
        for idx, (n_name, n_func) in enumerate(intra_nodes, 1):
            current = n_func(current, node_order=idx, use_analytics=use_analytics)
            output_saver.save_node_output(n_name, current, node_order=idx, table_name=table_name)
        state[table_name] = current  # update

    # --- inter-table phase
    cross_nodes = [
        ("merge_synonym_entities", merge_synonym_entities_node),
        ("merge_relation_types", merge_relation_types_node),
    ]
    for idx, (n_name, n_func) in enumerate(cross_nodes, 1):
        real_idx = idx + len(intra_nodes)
        state = n_func(state, node_order=real_idx, use_analytics=use_analytics)
        output_saver.save_node_output(n_name, state, node_order=real_idx, table_name="inter_table")
    return state
