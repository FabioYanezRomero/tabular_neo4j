"""
Header inference module for the Tabular to Neo4j converter.
This module handles inferring headers when none are detected.
"""

from typing import Dict, Any, List
import pandas as pd
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.csv_utils import df_to_json_sample
from Tabular_to_Neo4j.utils.metadata_utils import (
    get_metadata_for_state,
    format_metadata_for_prompt,
)
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def infer_header_llm_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to infer appropriate headers when no header is detected.

    Args:
        state: The current graph state
        config: LangGraph runnable configuration

    Returns:
        Updated graph state with inferred_header and final_header
    """
    if state.get("raw_dataframe") is None:
        error_msg = "Cannot infer header: no raw dataframe available"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        return state

    logger.info("Using LLM to infer headers for the CSV file")

    try:
        df = state["raw_dataframe"]

        # Get a sample of the data for the LLM
        sample_rows = min(MAX_SAMPLE_ROWS, len(df))
        data_sample = df_to_json_sample(df, sample_rows)

        # Get file name for the prompt
        file_name = os.path.basename(state.get("csv_file_path", "unknown.csv"))

        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = (
            format_metadata_for_prompt(metadata)
            if metadata
            else "No metadata available."
        )

        # Format the prompt with the data sample and metadata
        prompt = format_prompt(
            "infer_header.txt",
            metadata_text=metadata_text,
            num_columns=len(df.columns),
            data_sample=data_sample,
        )

        # Call the LLM to infer headers
        logger.debug("Calling LLM for header inference")
        response = call_llm_with_json_output(prompt, state_name="infer_header")

        # Extract the inferred headers
        inferred_header = response

        # Validate the response
        if not isinstance(inferred_header, list):
            error_msg = f"LLM did not return a list of headers: {inferred_header}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            return state

        if len(inferred_header) != len(df.columns):
            error_msg = f"LLM returned {len(inferred_header)} headers, but CSV has {len(df.columns)} columns"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            return state

        # Update the state with the inferred headers
        logger.info(f"Successfully inferred headers: {inferred_header}")
        state["inferred_header"] = inferred_header
        state["final_header"] = inferred_header

    except Exception as e:
        error_msg = f"Error inferring headers: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)

    return state
