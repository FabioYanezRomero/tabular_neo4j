"""Node that requests the LLM to map *all* columns of a table to graph elements in one shot.

Uses the `map_table_columns_to_graph_elements.txt` prompt. Stores output in
`state['column_graph_mapping']` (overwriting any per-column mapping) for consistency.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows, df_to_json_sample

logger = logging.getLogger(__name__)


def _columns_analytics_multiline(state: GraphState) -> str:
    analytics: Dict[str, Dict[str, Any]] = state.get("column_analytics", {}) or {}
    lines: List[str] = []
    for col, stats in analytics.items():
        lines.append(
            f"{col} | {stats.get('data_type', 'unknown')} | {stats.get('uniqueness_ratio', 0):.3f} | "
            f"{stats.get('cardinality', 0)} | {stats.get('missing_percentage', 0):.3f}"
        )
    return "\n".join(lines)


def map_table_columns_to_graph_elements_node(state: GraphState, node_order: int, use_analytics: bool = False) -> GraphState:
    logger.info("[map_table_columns_to_graph_elements_node] Starting")

    if state.get("column_analytics") is None:
        logger.warning("No column analytics – skipping table-level column mapping")
        return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))

    ent_det = state.get("table_entity_detection", {}) or {}
    intra_rel = state.get("intra_table_entity_relations", {}) or {}

    entities = ent_det.get("entities", []) if ent_det.get("has_entities") else []
    relationships = []
    if isinstance(intra_rel, dict):
        relationships = [rel.get("relationship_type") for rel in intra_rel.get("entity_relationships", [])]

    if not entities:
        logger.warning("No entities detected – column mapping may be meaningless; proceeding anyway")

    import os
    table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "table")))[0]

    sample_rows_json = "[]"
    if state.get("processed_dataframe") is not None:
        df_sample = get_sample_rows(state["processed_dataframe"], 5)
        if df_sample is not None and not df_sample.empty:
            sample_rows_json = df_to_json_sample(df_sample)

    prompt = format_prompt(
        template_name="map_table_columns_to_graph_elements.txt",
        table_name=table_name,
        entities=str(entities),
        relationships=str(relationships),
        columns_analytics=_columns_analytics_multiline(state),
        sample_rows=sample_rows_json,
        unique_suffix="",
        use_analytics=use_analytics
    )

    llm_resp = call_llm_with_json_output(
        prompt=prompt,
        state_name="map_table_columns_to_graph_elements",
        unique_suffix="",
        node_order=node_order,
        table_name=table_name,
        template_name="map_table_columns_to_graph_elements.txt",
    )

    if isinstance(llm_resp, dict) and "column_graph_mapping" in llm_resp:
        state["column_graph_mapping"] = llm_resp["column_graph_mapping"]
    else:
        state["column_graph_mapping"] = llm_resp  # raw response fallback

    logger.info("[map_table_columns_to_graph_elements_node] Completed")
    return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))
