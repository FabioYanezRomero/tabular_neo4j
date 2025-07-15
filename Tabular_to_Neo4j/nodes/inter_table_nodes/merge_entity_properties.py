"""Inter-table node: consolidate properties for entities after label merges.

Goal:
    After entity labels have been merged (e.g. Product -> Item) we may end up with
    instances of the preserved label that carry different sets of properties.  To
    simplify downstream schema generation we build, for every entity label, the
    union of all property names observed across *all* tables.

    The result is stored in the MultiTableGraphState under the key
    ``entity_property_unions`` as::

        {
            "entities": {
                "Item": ["itemId", "price", "product.name.tokens"],
                "Category": ["categoryId", ...],
                ...
            }
        }

    In addition, if desired, we can pad each table's ``column_graph_mapping`` so
    that missing properties are explicitly present (with ``added_by_merge`` flag)
    but, for now, we only compute the union – schema generators can rely on
    that.
"""
from __future__ import annotations

import logging
from typing import Dict, Set, Any, List

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState

logger = logging.getLogger(__name__)


def _collect_property_sets(state: MultiTableGraphState) -> Dict[str, Set[str]]:
    """Return mapping label -> set(property names) across all tables."""
    label_props: Dict[str, Set[str]] = {}
    for tbl_state in state.values():
        if not isinstance(tbl_state, GraphState):
            # Some global keys (e.g. relation_type_merges) live at root level
            # – skip them.
            continue
        col_map = tbl_state.get("column_graph_mapping", {}) or {}
        for prop_name, mapping in col_map.items():
            if not isinstance(mapping, dict):
                continue
            if mapping.get("graph_element_type") != "entity_property":
                continue
            label = mapping.get("belongs_to") or ""
            if not label:
                continue
            label_props.setdefault(label, set()).add(prop_name)
    return label_props


def merge_entity_properties_node(
    state: MultiTableGraphState, node_order: int, use_analytics: bool = False
):  # type: ignore[type-arg]
    """LangGraph node callable for property consolidation."""
    logger.info("[merge_entity_properties_node] Starting property-union consolidation")

    # ---------------- collect per-label property sets ----------------
    prop_union = _collect_property_sets(state)

    # ---------------- run LLM consolidation per merge group ---------
    merges_obj = state.get("entity_label_merges", {}) or {}
    merge_groups = merges_obj.get("merges", []) if isinstance(merges_obj, dict) else []
    consolidated: Dict[str, List[Dict[str, Any]]] = {}

    if not merge_groups:
        logger.warning("No entity label merges present – property consolidation skipped")
    else:
        from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
        from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output

        for idx, merge in enumerate(merge_groups, 1):
            target_labels = merge.get("to", []) if isinstance(merge.get("to"), list) else [merge.get("to")]
            source_labels = merge.get("from", []) if isinstance(merge.get("from"), list) else [merge.get("from")]
            # Only handle 1->1 merges for now – take first target label
            if not target_labels:
                continue
            target = target_labels[0]

            target_props = sorted(prop_union.get(target, []))
            source_props: Dict[str, List[str]] = {
                lbl: sorted(prop_union.get(lbl, [])) for lbl in source_labels
            }

            import json
            # Collect analytics for columns mapped to these labels
            column_analytics: Dict[str, Any] = {}
            if use_analytics:
                for tbl_state in state.values():
                    if not isinstance(tbl_state, GraphState):
                        continue
                    analytics = tbl_state.get("column_analytics", {}) or {}
                    col_map = tbl_state.get("column_graph_mapping", {}) or {}
                    for col_name, mapping in col_map.items():
                        if mapping.get("graph_element_type") != "entity_property":
                            continue
                        lbl = mapping.get("belongs_to") or ""
                        if lbl not in {target, *source_labels}:
                            continue
                        if col_name in analytics:
                            column_analytics[col_name] = analytics[col_name]
            # Compact analytics: keep only bool/int/float/str counts, drop sample rows
            def _compact(stats: Any) -> Any:
                if not isinstance(stats, dict):
                    return stats
                allowed = {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))}
                return allowed
            column_analytics = {k: _compact(v) for k, v in column_analytics.items()}

            prompt = format_prompt(
                "merge_entity_properties.txt",
                table_name="GLOBAL",
                target_label=target,
                target_properties=target_props,
                source_properties=source_props,
                column_analytics=json.dumps(column_analytics, indent=2),
                use_analytics=use_analytics,
            )
            llm_resp = call_llm_with_json_output(
                prompt=prompt,
                state_name="merge_entity_properties",
                unique_suffix=f"_{idx}",
                node_order=node_order,
                table_name="GLOBAL",
                template_name="merge_entity_properties.txt",
            )

            if not isinstance(llm_resp, dict) or "properties" not in llm_resp:
                logger.warning("Unexpected LLM response for property merge %d: %s", idx, llm_resp)
                continue

            consolidated[target] = llm_resp["properties"]

    # ---------------- store results back into state -----------------
    state["entity_property_unions"] = {"entities": consolidated if consolidated else {lbl: sorted(list(props)) for lbl, props in prop_union.items()}}
    logger.info(
        "[merge_entity_properties_node] Property consolidation completed for %d merge groups",
        len(consolidated),
    )

    return state
