"""High-level helpers for invoking language models."""

from typing import Dict, Any, Optional
import os

from Tabular_to_Neo4j.config.llm_providers import models
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.utils.llm_api import (
    get_model_info,
    list_loaded_models,
)
from Tabular_to_Neo4j.utils.prompt_utils import (
    format_prompt,
)
from Tabular_to_Neo4j.utils.output_saver import output_saver
from Tabular_to_Neo4j.utils.response_utils import extract_json_from_llm_response


__all__ = [
    "format_prompt",
    "call_llm_with_json_output",
    "get_model_info",
    "list_loaded_models",
]



logger = get_logger(__name__)

# Utility to get node order for consistent file prefixing, avoiding circular import
_node_order_map_cache = None


def call_llm_with_json_output(
    prompt: str,
    state_name: str = None,
    config: Optional[Dict[str, Any]] = None,
    unique_suffix: str = "",
    node_order: int = 0,
    table_name: str = None,
    template_name: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call the LLM and parse the response as JSON, using unified dispatcher and robust parsing.
    """
    from Tabular_to_Neo4j.config.settings import LLM_CONFIGS
    from Tabular_to_Neo4j.utils.llm_api import call_llm_api

    if state_name is None:
        state_name = "default"

    # Save original prompt for traceability
    try:
        if not output_saver:
            raise RuntimeError("OutputSaver is not initialized. All LLM output saving must use the same timestamp for the run.")
        base_dir = output_saver.base_dir
        timestamp = output_saver.timestamp
    except Exception as exc:
        logger.warning("Failed to save original prompt sample for state '%s': %s", state_name, exc)

    # Augment prompt to ask for JSON output
    json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
    state_config = LLM_CONFIGS.get(state_name, {})
    if config:
        state_config.update(config)

    try:
        response_text = call_llm_api(json_prompt, state_config)
        # logger.info(f"[LLM][{state_name}] Prompt: {json_prompt}")  # Removed to avoid printing prompt
        # logger.info(f"[LLM][{state_name}] Response: {response_text}")

        # --- Save raw LLM output for traceability ---
        
        import os, json
        # Save LLM output using OutputSaver's new method
        try:
            if output_saver:
                output_saver.save_llm_output_sample(
                    node_name=state_name,
                    output={
                        "prompt": json_prompt,
                        "response": response_text
                    },
                    node_order=node_order,
                    table_name=table_name,
                    unique_suffix=unique_suffix,
                    template_name=template_name,
                )
            else:
                logger.warning("OutputSaver is not initialized. Skipping LLM output saving.")
        except Exception as exc:
            logger.warning("Failed to save LLM output sample for state '%s': %s", state_name, exc)

        json_data = extract_json_from_llm_response(response_text)
        if json_data:
            return json_data
        logger.warning(f"Failed to parse JSON response for state '%s'. Response: %s", state_name, response_text)
        # Retry with stricter prompt
        retry_prompt = f"{prompt}\n\nYou MUST respond with ONLY valid JSON. No other text. No markdown formatting."
        retry_response_text = call_llm_api(retry_prompt, state_config)

        retry_json = extract_json_from_llm_response(retry_response_text)
        if retry_json:
            return retry_json
        logger.error(f"Failed to parse JSON from retry response for state '%s'. Response: %s", state_name, retry_response_text)
        return {}
    except Exception as exc:
        logger.error(f"Error invoking LLM for state '{state_name}': {exc}")
        return {"error": str(exc), "raw_response": ""}
