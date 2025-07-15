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


def _column_analytics_str(col_name: str, analytics: Any) -> str:
    """Return formatted analytics line.

    Works for both dict-style analytics and preformatted string analytics (contextualized).
    """
    if isinstance(analytics, str):
        # Already formatted upstream
        return analytics
    if not isinstance(analytics, dict):
        # Already formatted string or unknown type – return as-is
        return str(analytics)

    # Convert the full analytics dictionary to a compact JSON string so that **all**
    # computed analytics are available to the LLM prompt, not just a limited subset.
    try:
        return json.dumps(analytics, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback to default string conversion
        return str(analytics)


def map_column_to_graph_element_node(state: GraphState, node_order: int, use_analytics: bool = False) -> GraphState:
    logger.info("[map_column_to_graph_element_node] Starting")


    ent_det = state.get("table_entity_detection", {}) or {}
    intra_rel = state.get("intra_table_entity_relations", {}) or {}

    entities = ent_det.get("referenced_entities", []) if ent_det.get("has_entity_references") else []
    relationships = []
    if isinstance(intra_rel, dict):
        relationships = [rel.get("relationship_type") for rel in intra_rel.get("entity_relationships", [])]

    table_name = state.get("csv_file_path", "table").split("/")[-1].split(".")[0]

    col_results: Dict[str, Any] = {}

    for col in state.get("column_analytics", {}).keys():
        analytics = state.get("column_analytics", {}).get(col, {})
        # Sample values available only when analytics is dict
        sample_vals = analytics.get('sampled_values', []) if isinstance(analytics, dict) else []
        prompt = format_prompt(
            template_name="map_column_to_graph_element.txt",
            table_name=table_name,
            column_name=col,
            column_analytic=_column_analytics_str(col, analytics),
            entities=str(entities),
            relationships=str(relationships),
            sample_values=sample_vals,
            unique_suffix=col,
            use_analytics=use_analytics
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
