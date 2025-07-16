"""Graph pipeline using contextualized analytics (plain-text descriptions) instead of raw JSON analytics."""
from langgraph.graph import StateGraph, END
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.nodes.input import load_csv_node

from Tabular_to_Neo4j.nodes.Intra_table_nodes import (
    load_column_contextualized_node,
    detect_table_entities_node,
    infer_intra_table_relations_node,
    map_column_to_graph_element_node,
)

use_analytics = True  # we still have analytics-like info, but in text form

PIPELINE_NODES = [
    ("load_csv", load_csv_node),
    ("load_column_contextualized", load_column_contextualized_node),
    ("detect_table_entities", detect_table_entities_node),
    ("infer_intra_table_relations", infer_intra_table_relations_node),
    ("map_column_to_graph_element", map_column_to_graph_element_node),
]

from typing import Union, Tuple, Dict, Any
PipeEdge = Union[Tuple[str, str], Dict[str, Any]]

PIPELINE_EDGES: list[PipeEdge] = [
    ("load_csv", "load_column_contextualized"),
    ("load_column_contextualized", "detect_table_entities"),
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
        for idx, (name, func) in enumerate(PIPELINE_NODES, 1):
            def _make_node(f, order):
                def _node(state: GraphState):  # type: ignore[override]
                    extra_kwargs = {}
                    if f.__name__ == "detect_table_entities_node":
                        extra_kwargs["contextualized"] = True
                    return f(state, node_order=order, use_analytics=use_analytics, **extra_kwargs)
                return _node
            graph.add_node(name, _make_node(func, idx))
        for edge in PIPELINE_EDGES:
            if isinstance(edge, dict):
                graph.add_conditional_edges(edge["source"], edge["condition"], edge["edges"])
            else:
                graph.add_edge(*edge)
        graph.set_entry_point(ENTRY_POINT)
        graph.set_finish_point("map_column_to_graph_element")
        from Tabular_to_Neo4j.utils.output_saver import output_saver
        output_saver.set_node_order_map(PIPELINE_NODES)
        return graph
    except Exception as e:
        logger.error("Graph construction error: %s", e)
        logger.error(traceback.format_exc())
        raise

# ---------------- Multi-table runner -----------------
from typing import Dict, Any, Optional
import os
from Tabular_to_Neo4j.app_state import MultiTableGraphState
from Tabular_to_Neo4j.nodes.inter_table_nodes import (
    merge_synonym_entities_node,
    merge_entities_analytics_node,
    merge_relation_types_node,
)
from Tabular_to_Neo4j.nodes.inter_table_nodes.merge_entity_properties import merge_entity_properties_node
from Tabular_to_Neo4j.utils.output_saver import output_saver
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_path_for_csv

def run_column_map_multi_table_pipeline(table_folder: str, config: Optional[Dict[str, Any]] = None, use_analytics: bool = use_analytics) -> MultiTableGraphState:  # noqa: D401
    """Execute intra-table graph for each CSV in *table_folder* then run cross-table nodes."""
    import logging
    logger = logging.getLogger(__name__)

    if not output_saver:
        raise RuntimeError("OutputSaver not initialised – call init_output_saver() first")

    state: MultiTableGraphState = MultiTableGraphState()
    for fname in os.listdir(table_folder):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(table_folder, fname)
            meta_path = get_metadata_path_for_csv(csv_path)
            state[table_name] = GraphState(csv_file_path=csv_path, metadata_file_path=meta_path)

    intra_nodes = PIPELINE_NODES
    for table_name, tbl_state in state.items():
        current = tbl_state
        for idx, (n_name, n_func) in enumerate(intra_nodes, 1):
            try:
                extra_kwargs = {}
                if n_func.__name__ == "detect_table_entities_node":
                    extra_kwargs["contextualized"] = True
                current = n_func(current, node_order=idx, use_analytics=use_analytics, **extra_kwargs)
            except Exception as e:
                current = n_func(current, node_order=idx, **({"contextualized": True} if n_func.__name__ == "detect_table_entities_node" else {}))
            output_saver.save_node_output(n_name, current, node_order=idx, table_name=table_name)
        state[table_name] = current

    cross_nodes = [
        ("merge_synonym_entities", merge_synonym_entities_node),
        ("merge_entities_analytics", merge_entities_analytics_node),
        ("merge_relation_types", merge_relation_types_node),
        ("merge_entity_properties", merge_entity_properties_node),
    ]
    for idx, (n_name, n_func) in enumerate(cross_nodes, 1):
        real_idx = idx + len(intra_nodes)
        state = n_func(state, node_order=real_idx, use_analytics=use_analytics)
        output_saver.save_node_output(n_name, state, node_order=real_idx, table_name="inter_table")

    # --- persist final consolidated state (mirrors analytics pipeline) ---
    dataset_names = {"GLOBAL"}
    import json
    from Tabular_to_Neo4j.utils.serialization import json_default
    for ds in dataset_names:
        ds_dir = os.path.join(output_saver.output_dir, ds, "GLOBAL")
        os.makedirs(ds_dir, exist_ok=True)
        out_path = os.path.join(ds_dir, "final_state.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=json_default)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Could not write final_state.json: %s", e)
    return state
