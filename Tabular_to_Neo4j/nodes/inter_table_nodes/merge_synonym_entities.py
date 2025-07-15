"""Inter-table node that detects synonym entity labels across tables.

It collects all entity labels stored per-table in their respective `table_entity_detection`
state, prompts an LLM with `merge_synonym_entities.txt`, and stores a merge mapping
under the *global* state key `entity_label_merges`.

Expected upstream: state is a `MultiTableGraphState` where each value is a
`GraphState` that already contains `table_entity_detection` results.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Set

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output

logger = logging.getLogger(__name__)


def _collect_entity_labels(multi_state: MultiTableGraphState) -> List[str]:
    labels: Set[str] = set()
    for tbl_state in multi_state.values():
        if not isinstance(tbl_state, (GraphState, dict)):
            continue
        ted = tbl_state._extra.get("table_entity_detection", {}) or {}
        # Check for new format first
        if ted.get("has_entity_references", False):
            entities = ted.get("referenced_entities", [])
            if isinstance(entities, list):
                labels.update(entities)
    return sorted(labels)


def merge_synonym_entities_node(state: MultiTableGraphState, node_order: int, use_analytics: bool = False) -> MultiTableGraphState:  # type: ignore[type-arg]
    logger.info("[merge_synonym_entities_node] Starting synonym merge detection")

    entity_labels = _collect_entity_labels(state)
    if not entity_labels:
        logger.warning("No entity labels found across tables; skipping synonym merge detection")
        state["entity_label_merges"] = {"merges": []}
        return state

    prompt = format_prompt(
        template_name="merge_synonym_entities.txt",
        table_name="GLOBAL",
        entity_labels="\n".join(entity_labels),
        unique_suffix="",
        use_analytics=use_analytics,
    )

    llm_resp = call_llm_with_json_output(
        prompt=prompt,
        state_name="merge_synonym_entities",
        unique_suffix="",
        node_order=node_order,
        table_name="GLOBAL",
        template_name="merge_synonym_entities.txt",
    )

    # Basic validation
    if not isinstance(llm_resp, dict) or "merges" not in llm_resp:
        logger.error("LLM response missing 'merges'; storing raw response")
        merges = {"merges": llm_resp}
    else:
        merges = llm_resp

    state["entity_label_merges"] = merges
    logger.info("[merge_synonym_entities_node] Detected %d merge groups", len(merges.get("merges", [])))
    return state
