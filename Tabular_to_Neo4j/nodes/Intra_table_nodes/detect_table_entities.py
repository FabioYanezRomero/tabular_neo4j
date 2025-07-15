"""Node to detect whether a given table contains entity records and, if so, which entity labels.

This node is intended to be executed *per table* in the first LangGraph state. It formats
an LLM prompt defined in `prompts/detect_table_entities.txt`, invokes the Ollama/LangChain
LLM through `call_llm_with_json_output`, and stores the JSON response in the
`GraphState` under the key `table_entity_detection`.
"""
from __future__ import annotations

import logging
import json
from typing import Dict, Any

from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows, df_to_json_sample

logger = logging.getLogger(__name__)


def _build_columns_analytics(state: GraphState, use_analytics: bool = False, contextualized: bool = False) -> str:
    """Return multiline string summarising column analytics for the prompt."""
    analytics: Dict[str, Dict[str, Any]] = state.get("column_analytics", {}) or {}
    indent = "  "  # two spaces to match template indentation within code block
    lines = []
    if use_analytics:
        if contextualized:
            for col, stats in analytics.items():
                # The stats represent the string already formated
                line = stats
                lines.append(line)
        else:
            for col, stats in analytics.items():
                try:
                    json_str = json.dumps(stats, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    json_str = str(stats)
                line = f"{indent}{col} | {json_str}"
                lines.append(line)
        return "\n".join(lines)
    else:
        # return only the samples values
        for col, stats in analytics.items():
            line = (
                f"{indent}{col} | {stats.get('sampled_values', 'unknown')}"
            )
            lines.append(line)
        return "\n".join(lines)

def _get_sample_rows_json(state: GraphState, max_rows: int = 5) -> str:
    """Return up-to-`max_rows` sample rows from the *raw* DataFrame as JSON string.

    We purposefully avoid `processed_dataframe` here so that the node does not
    depend on downstream cleaning/pre-processing. If a raw_dataframe is not
    present (should not happen), we simply omit the sample rows.
    """
    if state.get("raw_dataframe") is None:
        return "[]"
    df_sample = get_sample_rows(state["raw_dataframe"], max_rows)
    if df_sample is None or df_sample.empty:
        return "[]"
    return df_to_json_sample(df_sample)


def detect_table_entities_node(state: GraphState, node_order: int, use_analytics: bool = False, *, contextualized: bool = False) -> GraphState:
    """Main entrypoint for the LangGraph node."""
    logger.info("[detect_table_entities_node] Starting entity detection for table")

    # Ensure prerequisites for prompt exist – we must have column analytics and some dataframe to sample rows from.
    if state.get("column_analytics") is None:
        msg = "Missing column analytics – skipping entity detection"
        logger.warning(msg)
        state.setdefault("error_messages", []).append(msg)
        return state if isinstance(state, GraphState) else GraphState.from_dict(dict(state))




    # Build prompt parts
    import os
    table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "table")))[0]
    columns_analytics_str = _build_columns_analytics(state, use_analytics, contextualized=contextualized)

    prompt = format_prompt(
        template_name="detect_table_entities.txt",
        table_name=table_name,
        columns_analytics=columns_analytics_str,
        unique_suffix="",
        use_analytics=use_analytics,
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
