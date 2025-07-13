"""Graph that operates on MultiTableGraphState to perform cross-table consolidation.

It expects that each value in the state dict is a populated GraphState from the
intra-table pipeline. The graph then chains all inter-table nodes:
1. merge_synonym_entities_node
2. merge_relation_types_node
3. column_semantic_similarity_node
4. infer_cross_table_relations_node
"""
from langgraph.graph import StateGraph, END
from Tabular_to_Neo4j.app_state import MultiTableGraphState
from Tabular_to_Neo4j.nodes.inter_table_nodes import (
    merge_synonym_entities_node,
    merge_relation_types_node,
    column_semantic_similarity_node,
    infer_cross_table_relations_node,
)

PIPELINE_NODES = [
    ("merge_synonym_entities", merge_synonym_entities_node),
    ("merge_relation_types", merge_relation_types_node),
    ("column_semantic_similarity", column_semantic_similarity_node),
    ("infer_cross_table_relations", infer_cross_table_relations_node),
]

PIPELINE_EDGES = [
    ("merge_synonym_entities", "merge_relation_types"),
    ("merge_relation_types", "column_semantic_similarity"),
    ("column_semantic_similarity", "infer_cross_table_relations"),
    ("infer_cross_table_relations", END),
]

ENTRY_POINT = PIPELINE_NODES[0][0]


def create_inter_table_graph() -> StateGraph:
    import logging, traceback
    logger = logging.getLogger(__name__)
    try:
        graph = StateGraph(MultiTableGraphState)
        for name, func in PIPELINE_NODES:
            graph.add_node(name, func)
        for edge in PIPELINE_EDGES:
            graph.add_edge(*edge)

        graph.set_entry_point(ENTRY_POINT)
        graph.set_finish_point("infer_cross_table_relations")

        from Tabular_to_Neo4j.utils.output_saver import output_saver
        output_saver.set_node_order_map(PIPELINE_NODES)
        return graph
    except Exception as e:
        logger.error("Inter-table graph construction error: %s", e)
        logger.error(traceback.format_exc())
        raise
