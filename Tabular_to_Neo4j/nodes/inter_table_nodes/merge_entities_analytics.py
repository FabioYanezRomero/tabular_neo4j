"""Inter-table node that validates proposed entity merges using column analytics.

It takes the `entity_label_merges` from the state, and for each proposed merge
group, it uses the `column_analytics` from each table to determine if the merge
is semantically sound. Merges deemed inadequate are removed or modified.

Stores results in `state['entity_label_merges']`.
"""
from __future__ import annotations

import logging
import json
from typing import Dict, Any, List, Set

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output

logger = logging.getLogger(__name__)

def _collect_all_column_analytics(state: MultiTableGraphState) -> Dict[str, Dict[str, Any]]:
    """Collects all column analytics from all tables in the state."""
    all_analytics: Dict[str, Dict[str, Any]] = {}
    for table_name, tbl_state in state.items():
        if isinstance(tbl_state, GraphState):
            analytics = tbl_state.get("column_analytics", {}) or {}
            for col_name, col_data in analytics.items():
                all_analytics[f"{table_name}.{col_name}"] = col_data
    return all_analytics

def merge_entities_analytics_node(state: MultiTableGraphState, node_order: int, use_analytics: bool = False) -> MultiTableGraphState:
    logger.info("[merge_entities_analytics_node] Starting entity merge validation using analytics")

    # Get proposed merges from the state
    proposed_merges = state.get("entity_label_merges", {}).get("merges", [])
    if not proposed_merges:
        logger.warning("No proposed entity merges found; skipping analytics validation.")
        return state

    # Collect all column analytics
    all_column_analytics = _collect_all_column_analytics(state)

    # Format the prompt for the LLM
    prompt = format_prompt(
        template_name="merge_entities_analytics.txt",
        proposed_merges=json.dumps(proposed_merges, indent=2),
        column_analytics=json.dumps(all_column_analytics, indent=2),
        unique_suffix="",
        use_analytics=use_analytics,
    )

    # Call the LLM
    llm_resp = call_llm_with_json_output(
        prompt=prompt,
        state_name="merge_entities_analytics",
        unique_suffix="",
        node_order=node_order,
        table_name="GLOBAL",
        template_name="merge_entities_analytics.txt",
    )

    # Validate and store the LLM response
    if not isinstance(llm_resp, dict) or "validated_merges" not in llm_resp:
        logger.error("LLM response missing 'validated_merges'; storing raw response")
        validated_merges = {"validated_merges": llm_resp}
    else:
        validated_merges = llm_resp

    state["entity_label_merges"] = validated_merges
    logger.info("[merge_entities_analytics_node] Validated %d merge groups", len(validated_merges.get("validated_merges", [])))

    return state
