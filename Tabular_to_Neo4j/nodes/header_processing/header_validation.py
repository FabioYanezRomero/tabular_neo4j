"""
Header validation module for the Tabular to Neo4j converter.
This module handles validating and improving headers.
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


def validate_header_llm_node(state: GraphState, node_order: int, use_analytics: bool = False) -> GraphState:
    """
    Use LLM to validate and potentially improve headers.

    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline

    Returns:
        Updated graph state with validated_header and potentially updated final_header
    """
    if state.get("raw_dataframe") is None or state.get("final_header") is None:
        error_msg = "Cannot validate header: missing raw dataframe or header"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        return state

    logger.info("Using LLM to validate and improve headers")

    try:
        df = state["raw_dataframe"]
        current_header = state["final_header"]

        # Get a sample of the data for the LLM
        sample_rows = min(MAX_SAMPLE_ROWS, len(df))
        data_sample = df_to_json_sample(df, sample_rows)

        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = (
            format_metadata_for_prompt(metadata)
            if metadata
            else "No metadata available."
        )

        # Log the header and metadata for debugging
        logger.debug(f"Current header: {current_header}")
        logger.debug(f"Metadata: {metadata}")

        # Set default values if validation fails
        state["validated_header"] = (
            current_header.copy()
            if isinstance(current_header, list)
            else current_header
        )

        # Extract language from metadata if available
        metadata_language = (
            metadata.get("language", "english") if metadata else "english"
        )
        logger.info(f"Metadata language: {metadata_language}")

        # Store metadata language in state for language detection node
        if "metadata_language" not in state:
            state["metadata_language"] = metadata_language
            state["metadata_language_confidence"] = (
                1.0  # High confidence since it's from metadata
            )

        import json

        headers_json = json.dumps(current_header)
        table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "")))[0]

        prompt = format_prompt(
            "validate_header.txt",
            table_name=table_name,
            data_sample=data_sample,
            headers=headers_json,  
            column_count=len(df.columns),
            row_count=len(df),
            metadata_text=metadata_text,
            use_analytics=use_analytics
        )

        logger.debug("Calling LLM for header validation")
        response = call_llm_with_json_output(
            prompt=prompt,
            state_name="validate_header",
            node_order=node_order,
            table_name=table_name,
            template_name="validate_header.txt"
        )
        if isinstance(response, list):
            validated_header = response
        else:
            validated_header = response.get("validated_header", current_header)

        if not isinstance(validated_header, list):
            if isinstance(validated_header, str):
                try:
                    validated_header = [h.strip() for h in validated_header.split(",")]
                except Exception:
                    logger.warning(
                        f"Could not parse validated_header as list: {validated_header}"
                    )
                    validated_header = current_header
            else:
                error_msg = f"LLM did not return a list of headers: {validated_header}"
                logger.error(error_msg)
                state["error_messages"].append(error_msg)
                validated_header = current_header

        if len(validated_header) != len(df.columns):
            error_msg = f"LLM returned {len(validated_header)} headers, but CSV has {len(df.columns)} columns"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            return state

        state["validated_header"] = validated_header

        headers_changed = validated_header != current_header

        if headers_changed:
            logger.info(f"Updated headers: {validated_header}")
            state["final_header"] = validated_header
        else:
            logger.info("LLM confirmed headers are appropriate")

    except Exception as e:
        error_msg = f"Error validating headers: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)

    return state
