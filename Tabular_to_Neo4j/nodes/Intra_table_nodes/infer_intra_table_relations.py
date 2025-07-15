"""Node that infers relationships between multiple entities detected within a single table.

Runs only when `table_entity_detection` reports more than one entity label. It formats
`prompts/infer_intra_table_entity_relations.txt`, sends it to the LLM via
`call_llm_with_json_output`, and stores the parsed JSON under
`intra_table_entity_relations` in the `GraphState`.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows, df_to_json_sample

logger = logging.getLogger(__name__)


def _build_columns_analytics(state: GraphState) -> str:
    """Return a multiline string of column analytics.

    Handles both normal analytics dicts *and* the contextualised variant where
    the value is already a formatted string produced upstream. This keeps the
    function backward-compatible while avoiding AttributeError when `stats` is
    a string.
    """
    analytics: Dict[str, Any] = state.get("column_analytics", {}) or {}
    lines: List[str] = []
    for col, stats in analytics.items():
        if isinstance(stats, str):
            # Contextualised analytics already formatted by previous node
            lines.append(stats)
            continue
        if not isinstance(stats, dict):
            # Fallback – unexpected type, just stringify
            lines.append(str(stats))
            continue
        lines.append(
            f"{col} | {stats.get('data_type', 'unknown')} | {stats.get('uniqueness_ratio', 0):.3f} | "
            f"{stats.get('cardinality', 0)} | {stats.get('missing_percentage', 0):.3f}"
        )
    return "\n".join(lines)


def _sample_rows(state: GraphState, max_rows: int = 5) -> str:
    if state.get("processed_dataframe") is None:
        return "[]"
    df_sample = get_sample_rows(state["processed_dataframe"], max_rows)
    if df_sample is None or df_sample.empty:
        return "[]"
    return df_to_json_sample(df_sample)


def infer_intra_table_relations_node(state: GraphState, node_order: int, use_analytics: bool = False) -> GraphState:
    """Infer relationships among entities within the same table."""
    logger.info("[infer_intra_table_relations_node] Starting")

    # Guard: require entity detection output with >1 entity
    ent_det = state.get("table_entity_detection", {}) or {}
    if not ent_det.get("has_entity_references") or len(ent_det.get("referenced_entities", [])) < 2:
        logger.info("[infer_intra_table_relations_node] Less than two entities – skipping relation inference")
        state["intra_table_entity_relations"] = {"skipped": True, "reason": "<2 entities"}
        return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))

    entities = ent_det.get("referenced_entities", [])

    # Build prompt
    import os
    table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "table")))[0]
    prompt = format_prompt(
        template_name="infer_intra_table_entity_relations.txt",
        table_name=table_name,
        entities=str(entities),
        columns_analytics=_build_columns_analytics(state),
        sample_rows=_sample_rows(state),
        unique_suffix="",
        use_analytics=use_analytics
    )

    llm_resp = call_llm_with_json_output(
        prompt=prompt,
        state_name="infer_intra_table_relations",
        unique_suffix="",
        node_order=node_order,
        table_name=table_name,
        template_name="infer_intra_table_entity_relations.txt",
    )

    state["intra_table_entity_relations"] = llm_resp
    logger.info("[infer_intra_table_relations_node] Completed: %s", llm_resp)

    return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))
