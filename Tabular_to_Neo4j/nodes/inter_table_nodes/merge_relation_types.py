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
from typing import Dict, Any, List, Set, Tuple, Counter

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output

logger = logging.getLogger(__name__)


# ----------------------------- helpers -----------------------------

def _relation_property_counts(state: MultiTableGraphState) -> Dict[str, int]:
    """Return number of distinct properties seen for each relationship type across all tables."""
    counts: Dict[str, Set[str]] = {}
    for tbl_state in state.values():
        if not isinstance(tbl_state, (GraphState, dict)):
            continue
        rel_list: List[Dict[str, Any]] = []
        if tbl_state.get("intra_table_entity_relations"):
            rel_list = tbl_state["intra_table_entity_relations"].get("entity_relationships", [])
        elif tbl_state.get("entity_relationships"):
            rel_list = tbl_state["entity_relationships"]
        for rel in rel_list:
            rel_type = rel.get("relationship_type") or rel.get("type")
            if not rel_type:
                continue
            rel_type_norm = _normalise_rel_type(rel_type)
            props = rel.get("properties", {}) if isinstance(rel.get("properties"), dict) else {}
            counts.setdefault(rel_type_norm, set()).update(props.keys())
    return {k: len(v) for k, v in counts.items()}

def _build_entity_merge_map(merges: List[Dict[str, Any]]) -> Dict[str, str]:
    """Return dict mapping **original** label -> merged label."""
    mapping: Dict[str, str] = {}
    for merge in merges:
        to_label_raw = merge.get("to")[0]
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
    # normalise relationship type to lower snake_case and remove duplicates
    norm: Set[Tuple[str, str, str]] = set()
    for s, r, t in candidates:
        r_norm = _normalise_rel_type(r)
        # treat direction as undirected by sorting endpoints
        a, b = sorted([s, t])
        norm.add((a, r_norm, b))
    return list(norm)


def _candidate_lines(candidates: List[Tuple[str, str, str]]) -> str:
    lines = [f"{s} --({r})-> {t}" for s, r, t in candidates]
    return "\n".join(sorted(set(lines)))

def _normalise_rel_type(rel: str) -> str:
    """Convert to lower snake_case (basic normalisation)."""
    import re
    rel = rel.strip().lower().replace(" ", "_")
    rel = re.sub(r"[^a-z0-9_]+", "_", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel

# ----------------------------- main node -----------------------------

def merge_relation_types_node(state: MultiTableGraphState, node_order: int, use_analytics: bool = False):  # type: ignore[type-arg]
    logger.info("[merge_relation_types_node] Starting relation-type merge detection")

    merges_obj = state.get("entity_label_merges", {})
    merge_groups = merges_obj.get("merges", []) if isinstance(merges_obj, dict) else []
    if not merge_groups:
        logger.warning("No entity merges present – skipping relation-type consolidation")
        state["relation_type_merges"] = {"merges": []}
        return state

    merge_map = _build_entity_merge_map(merge_groups)
    candidate_triples = _collect_candidate_relationships(state, merge_map)
    # remove self-loop triples where source and target are identical
    candidate_triples = [t for t in candidate_triples if t[0].lower() != t[2].lower()]
    if not candidate_triples:
        logger.warning("No candidate relationships connecting merged entities (after self-loop filter); nothing to merge")
        state["relation_type_merges"] = {"merges": []}
        return state

    # group by undirected pair of merged labels
    pair_to_types: Dict[Tuple[str, str], Set[str]] = {}
    for src, rel_type, tgt in candidate_triples:
        a, b = sorted([src, tgt])  # undirected
        pair_to_types.setdefault((a, b), set()).add(rel_type)

    all_merges: List[Dict[str, Any]] = []
    used_from: Set[str] = set()
    prop_counts = _relation_property_counts(state)

    for idx_pair, ((src_lbl, tgt_lbl), rel_types) in enumerate(pair_to_types.items(), 1):
        candidate_lines = [f"{src_lbl} --({r})-> {tgt_lbl}" for r in sorted(rel_types)]

        prompt = format_prompt(
            template_name="merge_relation_types.txt",
            table_name="GLOBAL",
            merged_entities=json.dumps(merge_groups, ensure_ascii=False, indent=2),
            candidate_relationships="\n".join(candidate_lines),
            unique_suffix=f"{src_lbl}__{tgt_lbl}",
            use_analytics=use_analytics,
        )

        llm_resp = call_llm_with_json_output(
            prompt=prompt,
            state_name="merge_relation_types",
            unique_suffix=f"{src_lbl}__{tgt_lbl}",
            node_order=node_order + idx_pair,  # keep ordering unique
            table_name="GLOBAL",
            template_name="merge_relation_types.txt",
        )

        if not isinstance(llm_resp, dict) or "merges" not in llm_resp:
            logger.error("LLM response for pair %s-%s missing 'merges'", src_lbl, tgt_lbl)
            continue

        for m in llm_resp["merges"]:
            raw_from_list = m.get("from", [])
            from_list_norm = [_normalise_rel_type(x) for x in raw_from_list]
            to_norm = _normalise_rel_type(m.get("to", ""))
            # choose ordering of from_list by property richness desc
            from_list_sorted = sorted(from_list_norm, key=lambda r: prop_counts.get(r, 0), reverse=True)
            if any(f in used_from for f in from_list_sorted):
                continue
            used_from.update(from_list_sorted)
            all_merges.append({"from": from_list_sorted, "to": to_norm})

    state["relation_type_merges"] = {"merges": all_merges}
    logger.info("[merge_relation_types_node] Detected %d relation merge groups", len(all_merges))
    return state
