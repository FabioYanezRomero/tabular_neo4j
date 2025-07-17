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

    # Build a mapping of entity label -> {column_key: analytics} using prior column_graph_mapping decisions
    def _collect_entity_column_analytics(state: MultiTableGraphState) -> Dict[str, Dict[str, Any]]:
        """Return analytics grouped by the entity each column was mapped to."""
        entity_map: Dict[str, Dict[str, Any]] = {}
        for tbl_name, tbl_state in state.items():
            if not isinstance(tbl_state, GraphState):
                continue
            col_mapping: Dict[str, Any] = tbl_state.get("column_graph_mapping", {}) or {}
            analytics = tbl_state.get("column_analytics", {}) or {}
            for col_name, mapping_info in col_mapping.items():
                if not isinstance(mapping_info, dict):
                    continue
                if mapping_info.get("graph_element_type") != "entity_property":
                    continue
                entity_label = mapping_info.get("belongs_to") or ""
                if not entity_label:
                    continue
                col_key = f"{tbl_name}.{col_name}"
                col_stats = analytics.get(col_name)
                if col_stats is None:
                    continue
                entity_map.setdefault(entity_label, {})[col_key] = col_stats
        return entity_map

    entity_to_col_analytics = _collect_entity_column_analytics(state)

    # Get proposed merges from the state
    proposed_merges = state.get("entity_label_merges", {}).get("merges", [])
    if not proposed_merges:
        logger.warning("No proposed entity merges found; skipping analytics validation.")
        return state

    # Collect and filter column analytics so the prompt stays concise
    # Fallback analytics across all columns (used if entity-specific mapping empty)
    all_column_analytics = _collect_all_column_analytics(state)

    def _filter_analytics_for_merges(merges, analytics):
        """Return only analytics whose column key mentions any label in the proposed merges."""
        def _extract_labels(merge):
            from_field = merge.get("from", [])
            if isinstance(from_field, list):
                labels = from_field.copy()
            else:
                labels = [from_field]
            labels.append(merge.get("to", ""))
            return labels
        label_set = {lbl.lower() for m in merges for lbl in _extract_labels(m)}
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

    # Explode merges so each entry has exactly one 'from' label and one 'to' label.
    exploded: List[Dict[str, Any]] = []
    for merge in proposed_merges:
        # Skip malformed merge entries that are not dictionaries
        if not isinstance(merge, dict):
            logger.warning(f"Skipping malformed merge entry (expected dict but got {type(merge)}): {merge}")
            continue
        from_field = merge.get("from")
        to_label = merge.get("to")
        if isinstance(from_field, list):
            for lbl in from_field:
                if lbl.lower() == (to_label or "").lower():
                    # skip identity merge
                    continue
                new_merge = merge.copy()
                new_merge["from"] = lbl
                new_merge["to"] = to_label
                exploded.append(new_merge)
        else:
            exploded.append(merge)
    proposed_merges = exploded

    # Apply filtering and compaction
    all_column_analytics = _filter_analytics_for_merges(proposed_merges, all_column_analytics)
    all_column_analytics = {k: _compact_stats(v) for k, v in all_column_analytics.items()}

    # Iterate through each merge pair and validate individually to minimise prompt size
    validated_groups = []
    for idx, merge in enumerate(proposed_merges):
        # Prefer analytics only for columns previously mapped to the entities in this merge
        from_field = merge.get("from", [])
        if isinstance(from_field, list):
            lbls = from_field.copy()
        else:
            lbls = [from_field]
        lbls.append(merge.get("to", ""))
        labels_in_merge = set([lbl for lbl in lbls if lbl])
        single_analytics: Dict[str, Any] = {}
        for lbl in labels_in_merge:
            if lbl in entity_to_col_analytics:
                single_analytics[lbl] = {k: _compact_stats(v) for k, v in entity_to_col_analytics[lbl].items()}
        # If none found (e.g., mapping missing), fall back to heuristic filtering
        if not single_analytics:
            tmp = _filter_analytics_for_merges([merge], all_column_analytics)
            single_analytics = {k: _compact_stats(v) for k, v in tmp.items()}

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
