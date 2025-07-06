"""Inter-table node to consolidate relationship type names after entity label merges.

Assumes `entity_label_merges` exists in the global state (output of
`merge_synonym_entities_node`). It scans intra-table relationship lists in each
GraphState (`intra_table_entity_relations` or `entity_relationships`), selects
those connecting merged entities, and prompts an LLM to suggest unified relation
names.
"""
from __future__ import annotations

import logging
import json
from typing import Dict, Any, List, Set, Tuple

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output

logger = logging.getLogger(__name__)


# ----------------------------- helpers -----------------------------

def _build_entity_merge_map(merges: List[Dict[str, Any]]) -> Dict[str, str]:
    """Return dict mapping **original** label -> merged label."""
    mapping: Dict[str, str] = {}
    for merge in merges:
        to_label_raw = merge.get("to")
        if not isinstance(to_label_raw, str):
            logger.warning("Skipping merge entry with non-string 'to': %s", merge)
            continue
        to_label: str = to_label_raw

        for frm_raw in merge.get("from", []):
            if not isinstance(frm_raw, str):
                logger.warning("Skipping non-string 'from' value: %s in %s", frm_raw, merge)
                continue
            mapping[frm_raw] = to_label
    return mapping


def _collect_candidate_relationships(state: MultiTableGraphState, merge_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """Collect (src_label, rel_type, tgt_label) tuples where both labels participate in merge_map.
    If a label is not in merge_map, keep as-is but still consider.
    """
    candidates: List[Tuple[str, str, str]] = []
    for tbl_state in state.values():
        if not isinstance(tbl_state, (GraphState, dict)):
            continue
        rel_list: List[Dict[str, Any]] = []
        if tbl_state.get("intra_table_entity_relations"):
            rel_list = tbl_state["intra_table_entity_relations"].get("entity_relationships", [])
        elif tbl_state.get("entity_relationships"):
            rel_list = tbl_state["entity_relationships"]
        for rel in rel_list:
            src = rel.get("source_entity") or rel.get("source")
            tgt = rel.get("target_entity") or rel.get("target")
            rel_type = rel.get("relationship_type") or rel.get("type")
            if not (src and tgt and rel_type):
                continue
            # Check if both labels are in merge_map (either originally or as target merge)
            src_mapped = merge_map.get(src, src)
            tgt_mapped = merge_map.get(tgt, tgt)
            if src_mapped == tgt_mapped:
                # self relationships, still include
                pass
            # Consider only relationships where at least one of the endpoints was merged
            if (src in merge_map) or (tgt in merge_map):
                candidates.append((src_mapped, rel_type, tgt_mapped))
    return candidates


def _candidate_lines(candidates: List[Tuple[str, str, str]]) -> str:
    lines = [f"{s} --({r})-> {t}" for s, r, t in candidates]
    return "\n".join(sorted(set(lines)))

# ----------------------------- main node -----------------------------

def merge_relation_types_node(state: MultiTableGraphState, node_order: int):  # type: ignore[type-arg]
    logger.info("[merge_relation_types_node] Starting relation-type merge detection")

    merges_obj = state.get("entity_label_merges", {})
    merge_groups = merges_obj.get("merges", []) if isinstance(merges_obj, dict) else []
    if not merge_groups:
        logger.warning("No entity merges present – skipping relation-type consolidation")
        state["relation_type_merges"] = {"merges": []}
        return state

    merge_map = _build_entity_merge_map(merge_groups)
    candidates = _collect_candidate_relationships(state, merge_map)
    if not candidates:
        logger.warning("No candidate relationships connecting merged entities; nothing to merge")
        state["relation_type_merges"] = {"merges": []}
        return state

    prompt = format_prompt(
        template_name="merge_relation_types.txt",
        merged_entities=json.dumps(merge_groups, ensure_ascii=False, indent=2),
        candidate_relationships=_candidate_lines(candidates),
        unique_suffix="",
    )

    llm_resp = call_llm_with_json_output(
        prompt=prompt,
        state_name="merge_relation_types",
        unique_suffix="",
        node_order=node_order,
        table_name="GLOBAL",
        template_name="merge_relation_types.txt",
    )

    if not isinstance(llm_resp, dict) or "merges" not in llm_resp:
        logger.error("LLM response missing 'merges'; storing raw response")
        merges_out = {"merges": llm_resp}
    else:
        merges_out = llm_resp

    state["relation_type_merges"] = merges_out
    logger.info("[merge_relation_types_node] Detected %d relation merge groups", len(merges_out.get("merges", [])))
    return state
