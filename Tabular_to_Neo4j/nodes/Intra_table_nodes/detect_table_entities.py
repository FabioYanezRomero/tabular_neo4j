"""Node to detect whether a given table contains entity records and, if so, which entity labels.

This node is intended to be executed *per table* in the first LangGraph state. It formats
an LLM prompt defined in `prompts/detect_table_entities.txt`, invokes the Ollama/LangChain
LLM through `call_llm_with_json_output`, and stores the JSON response in the
`GraphState` under the key `table_entity_detection`.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows, df_to_json_sample

logger = logging.getLogger(__name__)


def _build_columns_analytics(state: GraphState) -> str:
    """Return multiline string summarising column analytics for the prompt."""
    analytics: Dict[str, Dict[str, Any]] = state.get("column_analytics", {}) or {}
    lines = []
    for col, stats in analytics.items():
        line = (
            f"{col} | {stats.get('data_type', 'unknown')} | "
            f"{stats.get('uniqueness_ratio', 0):.3f} | "
            f"{stats.get('cardinality', 0)} | "
            f"{stats.get('missing_percentage', 0):.3f}"
        )
        lines.append(line)
    return "\n".join(lines)


def _get_sample_rows_json(state: GraphState, max_rows: int = 5) -> str:
    """Return up-to-`max_rows` sample rows from the processed DataFrame as JSON string."""
    if state.get("processed_dataframe") is None:
        return "[]"
    df_sample = get_sample_rows(state["processed_dataframe"], max_rows)
    if df_sample is None or df_sample.empty:
        return "[]"
    return df_to_json_sample(df_sample)


def detect_table_entities_node(state: GraphState, node_order: int) -> GraphState:
    """Main entrypoint for the LangGraph node."""
    logger.info("[detect_table_entities_node] Starting entity detection for table")

    # Early validation of required state elements
    if state.get("column_analytics") is None or state.get("processed_dataframe") is None:
        msg = "Missing column analytics or processed dataframe — skipping entity detection"
        logger.warning(msg)
        state.setdefault("error_messages", []).append(msg)
        return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))

    # Build prompt parts
    import os
    table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "table")))[0]
    columns_analytics_str = _build_columns_analytics(state)
    sample_rows_json = _get_sample_rows_json(state)

    prompt = format_prompt(
        template_name="detect_table_entities.txt",
        table_name=table_name,
        columns_analytics=columns_analytics_str,
        sample_rows=sample_rows_json,
        unique_suffix="",
    )

    llm_response = call_llm_with_json_output(
        prompt=prompt,
        state_name="detect_table_entities",
        unique_suffix="",
        node_order=node_order,
        table_name=table_name,
        template_name="detect_table_entities.txt",
    )

    # Store in state
    state["table_entity_detection"] = llm_response
    logger.info("[detect_table_entities_node] Detection complete: %s", llm_response)

    return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))
