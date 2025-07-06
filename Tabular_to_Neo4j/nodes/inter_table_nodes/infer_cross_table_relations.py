"""Inter-table node to infer semantic relations between columns of different tables.

It reuses the existing prompt `infer_cross_table_column_relation.txt` and logic
from `cross_table_analysis.llm_relation_node` but adapts to the new similarity
format produced by `column_semantic_similarity_node` (a list of dicts with
`table_a`, `column_a`, `table_b`, `column_b`, `similarity`).

Stores results in `state['cross_table_inferred_relations']`.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output

logger = logging.getLogger(__name__)


SIM_THRESHOLD = 0.85  # keep consistent with similarity node


def _build_context_map(tbl_state: GraphState) -> Dict[str, str]:
    ctx = {}
    for item in tbl_state.get("columns_contextualization", []):
        ctx[item["column"]] = item["contextualization"]
    return ctx


def infer_cross_table_relations_node(state: MultiTableGraphState, node_order: int):  # type: ignore[type-arg]
    logger.info("[infer_cross_table_relations_node] Starting cross-table relation inference")

    similarity_pairs: List[Dict[str, Any]] = state.get("cross_table_column_similarity", [])  # global key
    if not similarity_pairs:
        logger.warning("No similarity pairs found; skipping relation inference")
        state["cross_table_inferred_relations"] = {}
        return state

    relations: Dict[str, Any] = {}

    for pair in similarity_pairs:
        sim = pair.get("similarity", 0)
        if sim < SIM_THRESHOLD:
            continue
        table1 = pair.get("table_a")
        col1 = pair.get("column_a")
        table2 = pair.get("table_b")
        col2 = pair.get("column_b")
        if not (table1 and table2 and col1 and col2):
            continue
        pair_key = f"{table1}.{col1} <-> {table2}.{col2}"

        ctx1_map = _build_context_map(state.get(table1, GraphState()))
        ctx2_map = _build_context_map(state.get(table2, GraphState()))

        prompt = format_prompt(
            "infer_cross_table_column_relation.txt",
            table_name="inter_table",
            col1=col1,
            table1=table1,
            col2=col2,
            table2=table2,
            context1=ctx1_map.get(col1, ""),
            context2=ctx2_map.get(col2, ""),
            similarity=sim,
        )

        llm_result = call_llm_with_json_output(
            prompt=prompt,
            state_name="infer_cross_table_relations",
            unique_suffix=f"{table1}_{col1}__{table2}_{col2}",
            node_order=node_order,
            table_name="inter_table",
            template_name="infer_cross_table_column_relation.txt",
        )
        relations[pair_key] = llm_result if isinstance(llm_result, dict) else {"raw_response": llm_result}

    state["cross_table_inferred_relations"] = relations
    logger.info("[infer_cross_table_relations_node] Stored %d inferred relations", len(relations))
    return state
