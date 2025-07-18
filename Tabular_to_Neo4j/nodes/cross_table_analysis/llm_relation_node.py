"""
Node for determining the semantic relation between similar columns across tables using an LLM (Ollama or LMStudio).
"""

from typing import Dict, Any, Optional
from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
import logging

logger = logging.getLogger(__name__)

PROMPT_PATH = "/app/Tabular_to_Neo4j/prompts/infer_cross_table_column_relation.txt"

def load_llm_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def llm_relation_node(state: MultiTableGraphState, node_order: int, use_analytics: bool = False) -> MultiTableGraphState:
    """
    For each pair of columns with similarity above threshold, call LLM to decide the relationship.
    Stores results in each table's GraphState under 'cross_table_column_relations'.
    """

    threshold = 0.85

    # Gather all pairs from the similarity matrix (should be present in each table's GraphState)
    relations = {}
    for table_name, table_state in state.items():
        similarity_matrix = table_state.get("cross_table_column_similarity", {})
        for pair_key, similarity in similarity_matrix.items():
            if similarity < threshold:
                continue
            # Parse pair_key: "table1.col1 <-> table2.col2"
            try:
                left, right = pair_key.split(" <-> ")
                table1, col1 = left.split(".", 1)
                table2, col2 = right.split(".", 1)
            except Exception as e:
                logger.warning(f"Failed to parse pair key '{pair_key}': {e}")
                continue
            # Get context for each column if available
            context1 = state.get(table1, {}).get("columns_contextualization", [])
            context2 = state.get(table2, {}).get("columns_contextualization", [])
            context1_map = {c["column"]: c["contextualization"] for c in context1}
            context2_map = {c["column"]: c["contextualization"] for c in context2}
            prompt = format_prompt(
                "infer_cross_table_column_relation.txt",
                table_name="inter_table",
                col1=col1,
                table1=table1,
                col2=col2,
                table2=table2,
                context1=context1_map.get(col1, ""),
                context2=context2_map.get(col2, ""),
                similarity=similarity,
                use_analytics=use_analytics,
            )
            # Call the LLM using the unified dispatcher with JSON output
            llm_result = call_llm_with_json_output(
                prompt=prompt,
                state_name="llm_relation_node",
                unique_suffix=f"{table1}_{col1}__{table2}_{col2}",
                node_order=node_order,
                table_name="inter_table",
                template_name="infer_cross_table_column_relation.txt",
            )
            relations[pair_key] = llm_result if isinstance(llm_result, dict) else {"raw_response": llm_result}
    import logging
    logger = logging.getLogger(__name__)
    logger.info('[llm_relation_node][BEFORE] Table states: ' + str({k: type(v).__name__ for k,v in state.items()}))
    # Store results in each table's GraphState

    for table_name, table_state in state.items():
        if not isinstance(table_state, GraphState):
            # Recover: wrap dict in GraphState
            table_state = GraphState(**table_state)
            state[table_name] = table_state
        table_state["cross_table_column_relations"] = {
            k: v for k, v in relations.items() if k.startswith(f"{table_name}.") or k.split(" <-> ")[1].startswith(f"{table_name}.")
        }
    # Ensure every table state is a GraphState before returning
    for table_name, table_state in state.items():
        if not isinstance(table_state, GraphState):
            state[table_name] = GraphState(**dict(table_state))
    logger.info('[llm_relation_node][AFTER] Table states: ' + str({k: type(v).__name__ for k,v in state.items()}))
    return state
