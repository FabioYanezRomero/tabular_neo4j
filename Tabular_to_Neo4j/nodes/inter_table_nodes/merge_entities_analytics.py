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

    # Collect and filter column analytics so the prompt stays concise
    all_column_analytics = _collect_all_column_analytics(state)

    def _filter_analytics_for_merges(merges, analytics):
        """Return only analytics whose column key mentions any label in the proposed merges."""
        label_set = {lbl.lower() for m in merges for lbl in m.get("from", []) + [m.get("to", "")]}
        return {
            col_key: stats
            for col_key, stats in analytics.items()
            if any(lbl in col_key.lower() for lbl in label_set)
        }

    def _compact_stats(stats: Any) -> Any:
        """Return compacted analytics dict or pass through if already a string.

        Contextualised analytics may be plain strings rather than dicts; in that
        case we forward them unchanged. If *stats* is neither dict nor str, we
        convert to ``str`` to avoid type errors.
        """
        if isinstance(stats, str):
            return stats
        if not isinstance(stats, dict):
            return str(stats)
        keep = (
            "data_type",
            "uniqueness_ratio",
            "cardinality",
            "missing_percentage",
            "min_value",
            "max_value",
        )
        return {k: stats[k] for k in keep if k in stats}

    # Eliminate duplicates in proposed merges
    proposed_merges_copy = []
    for merge in proposed_merges:
        from_labels_first = merge.get("from")[0]
        try:
            from_labels_second = merge.get("from")[1]
        except IndexError:
            continue
        to_label = merge.get("to")[0]
        if from_labels_first.lower() == from_labels_second.lower() and to_label.lower() == from_labels_first.lower():
            continue
        proposed_merges_copy.append(merge)

    proposed_merges = proposed_merges_copy

    # Apply filtering and compaction
    all_column_analytics = _filter_analytics_for_merges(proposed_merges, all_column_analytics)
    all_column_analytics = {k: _compact_stats(v) for k, v in all_column_analytics.items()}

    # Iterate through each merge pair and validate individually to minimise prompt size
    validated_groups = []
    for idx, merge in enumerate(proposed_merges):
        single_analytics = _filter_analytics_for_merges([merge], all_column_analytics)
        single_analytics = {k: _compact_stats(v) for k, v in single_analytics.items()}

        prompt = format_prompt(
            template_name="merge_entities_analytics.txt",
            table_name="GLOBAL",
            proposed_merges=json.dumps([merge], indent=2),
            column_analytics=json.dumps(single_analytics, indent=2),
            unique_suffix=f"_{idx}",
            use_analytics=use_analytics,
        )

        llm_resp = call_llm_with_json_output(
            prompt=prompt,
            state_name="merge_entities_analytics",
            unique_suffix=f"_{idx}",
            node_order=node_order,
            table_name="GLOBAL",
            template_name="merge_entities_analytics.txt",
        )

        if isinstance(llm_resp, dict) and "validated_merges" in llm_resp:
            validated_groups.extend(llm_resp["validated_merges"])
        else:
            logger.warning("Unexpected LLM response for merge index %d: %s", idx, llm_resp)

        # Normalise validated merge objects to a standard schema expected downstream
    normalised = []
    for m in validated_groups:
        if not isinstance(m, dict):
            continue
        if "from" in m and "to" in m:
            from_labels = m["from"] if isinstance(m["from"], list) else [m["from"]]
            to_labels = m["to"] if isinstance(m["to"], list) else [m["to"]]
        elif "source" in m and "target" in m:
            from_labels = [m["source"]]
            to_labels = [m["target"]]
        else:
            # Unrecognised format, skip
            logger.warning("Skipping merge with unrecognised keys: %s", m)
            continue
        normalised.append({
            "from": from_labels,
            "to": to_labels,
            "confidence": m.get("confidence"),
            "reasoning": m.get("reasoning"),
        })

    # Store aggregated and normalised merges (legacy key name 'merges')
    state["entity_label_merges"] = {"merges": normalised}
    logger.info("[merge_entities_analytics_node] Validated %d merge groups", len(normalised))

    return state
