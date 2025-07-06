"""Node that maps a single column to an entity or relationship graph element.

Although the prompt is *per column*, this node loops over every column in the
current table, calls the LLM once per column with the
`map_column_to_graph_element.txt` prompt, and accumulates the results in
`state['column_graph_mapping']`.
"""
from __future__ import annotations

import logging
import json
from typing import Dict, Any, List

from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows

logger = logging.getLogger(__name__)


def _column_analytics_str(col_name: str, analytics: Dict[str, Any]) -> str:
    return (
        f"{col_name} | {analytics.get('data_type', 'unknown')} | "
        f"{analytics.get('uniqueness_ratio', 0):.3f} | "
        f"{analytics.get('cardinality', 0)} | "
        f"{analytics.get('missing_percentage', 0):.3f}"
    )


def map_column_to_graph_element_node(state: GraphState, node_order: int) -> GraphState:
    logger.info("[map_column_to_graph_element_node] Starting")


    ent_det = state.get("table_entity_detection", {}) or {}
    intra_rel = state.get("intra_table_entity_relations", {}) or {}

    entities = ent_det.get("entities", []) if ent_det.get("has_entities") else []
    relationships = []
    if isinstance(intra_rel, dict):
        relationships = [rel.get("relationship_type") for rel in intra_rel.get("entity_relationships", [])]

    table_name = state.get("csv_file_path", "table").split("/")[-1].split(".")[0]

    col_results: Dict[str, Any] = {}

    for col in state.get("final_header", []) or []:
        analytics = state.get("column_analytics", {}).get(col, {})
        sample_vals = []
        if state.get("processed_dataframe") is not None and col in state["processed_dataframe"].columns:
            sample_vals = state["processed_dataframe"][col].head(5).tolist()
        prompt = format_prompt(
            template_name="map_column_to_graph_element.txt",
            table_name=table_name,
            column_name=col,
            column_analytic=_column_analytics_str(col, analytics),
            entities=str(entities),
            relationships=str(relationships),
            sample_values=json.dumps(sample_vals),
            unique_suffix=col,
        )
        llm_resp = call_llm_with_json_output(
            prompt=prompt,
            state_name="map_column_to_graph_element",
            unique_suffix=col,
            node_order=node_order,
            table_name=table_name,
            template_name="map_column_to_graph_element.txt",
        )
        col_results[col] = llm_resp

    state["column_graph_mapping"] = col_results
    logger.info("[map_column_to_graph_element_node] Completed for %d columns", len(col_results))
    return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))
