"""
Header translation module for the Tabular to Neo4j converter.
This module handles translating headers to match the metadata language.
"""

from typing import Dict, Any, List
import pandas as pd
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import (
    get_metadata_for_state,
    format_metadata_for_prompt,
)
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def translate_header_llm_node(state: GraphState, node_order: int, use_analytics: bool = False) -> GraphState:
    """
    Use LLM to translate headers to match the metadata language.
    This node is only called if the detect_header_language_node determined translation is needed.

    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline

    Returns:
        Updated graph state with translated_header and potentially updated final_header
    """
    if state.get("final_header") is None:
        error_msg = "Cannot translate header: no header available"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        return state

    # Get the header language and metadata language
    header_language = state.get("header_language", "Unknown")
    metadata_language = state.get("metadata_language")  # Set by language detection

    logger.info(
        f"Using LLM to translate headers from {header_language} to {metadata_language}"
    )

    try:
        header = state["final_header"]

        # Get file name for the prompt
        file_name = os.path.basename(state.get("csv_file_path", "unknown.csv"))

        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = (
            format_metadata_for_prompt(metadata)
            if metadata
            else "No metadata available."
        )

        # Extract table_name before formatting prompt
        # Extract table_name from csv_file_path if possible

        table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "")))[0]
        template_name = "translate_header.txt"
        # Format the prompt with the header, language information, and metadata
        prompt = format_prompt(
            template_name=template_name,
            table_name=table_name,
            subfolder="header_translation",
            file_name=file_name,
            headers=header,
            source_language=header_language,
            target_language=metadata_language,
            metadata_text=metadata_text,
            use_analytics=use_analytics
        )

        # Call the LLM to translate headers
        logger.debug("Calling LLM for header translation")
        response = call_llm_with_json_output(
            prompt,
            table_name=table_name,
            template_name=template_name,
            state_name="translate_header",
            node_order=node_order,
            is_translation=True,
        )

        # Extract the translated headers
        translated_header = response

        # Validate the response
        if not isinstance(translated_header, list):
            # If the response is a dict indicating no translation was needed, skip error
            if isinstance(translated_header, dict) and translated_header.get("is_in_target_language") is True:
                logger.info("Header is already in the target language; no translation needed.")
                return state
            error_msg = (
                f"LLM did not return a list of translated headers: {translated_header}"
            )
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            return state

        if len(translated_header) != len(header):
            error_msg = f"LLM returned {len(translated_header)} headers, but original has {len(header)} columns"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            return state

        # Update the state with the translated headers
        logger.info(
            f"Successfully translated headers from {header_language} to {metadata_language}"
        )
        logger.debug(f"Original headers: {header}")
        logger.debug(f"Translated headers: {translated_header}")

        state["translated_header"] = translated_header
        state["final_header"] = translated_header

    except Exception as e:
        error_msg = f"Error translating headers: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)

    return state
