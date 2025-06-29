from typing import Any
import logging
import os
from langgraph.graph import StateGraph
logger = logging.getLogger(__name__)

def create_graph(pipeline: str = "single_table_graph") -> 'StateGraph':
    """
    Selects the appropriate workflow graph based on the pipeline argument.
    Only used for single-table pipeline.
    """
    if pipeline == "single_table_graph":
        from Tabular_to_Neo4j.graphs.single_table_graph import create_single_table_graph
        return create_single_table_graph()
    elif pipeline == "multi_table_graph":
        from Tabular_to_Neo4j.graphs.multi_table_graph import create_multi_table_graph
        return create_multi_table_graph()
    else:
        logger.warning(f"Unknown pipeline '{pipeline}', defaulting to single_table_graph.")
        from Tabular_to_Neo4j.graphs.single_table_graph import create_single_table_graph
        return create_single_table_graph()

def validate_input_path(input_path: str, pipeline: str):
    if pipeline == "single_table_graph":
        if not os.path.isfile(input_path):
            logger.error(f"Expected a CSV file for single-table pipeline, got: {input_path}")
            raise ValueError("Input must be a CSV file for single-table pipeline.")
    elif pipeline == "multi_table_graph":
        if not os.path.isdir(input_path):
            logger.error(f"Expected a directory of CSVs for multi-table pipeline, got: {input_path}")
            raise ValueError("Input must be a directory for multi-table pipeline.")
    else:
        logger.error(f"Unknown pipeline: {pipeline}")
        raise ValueError(f"Unknown pipeline: {pipeline}")