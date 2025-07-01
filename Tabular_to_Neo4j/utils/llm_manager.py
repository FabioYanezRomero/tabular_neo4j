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

    try:
        response_text = call_llm_api(json_prompt, state_config)
        
        import os, json
        try:
            def _clean_llm_output(text):
                import re
                if not isinstance(text, str):
                    return text
                text = re.sub(r"^```(?:json)?\\s*", "", text.strip())
                text = re.sub(r"```$", "", text.strip())
                text = text.encode('utf-8').decode('unicode_escape')
                text = re.sub(r'\n+', '\n', text)
                text = text.strip('\n')
                return text

            cleaned_response = _clean_llm_output(response_text)
            parsed_response = extract_json_from_llm_response(cleaned_response)
            output_dict = {
                "response": parsed_response if parsed_response else cleaned_response
            }
            output_saver.save_llm_output_sample(
                node_name=state_name,
                output=output_dict,
                node_order=node_order,
                table_name=table_name,
                unique_suffix=unique_suffix,
                template_name=template_name,
            )
        except Exception as exc:
            logger.warning("Failed to save LLM output sample for state '%s': %s", state_name, exc)
        
        if parsed_response:
            return parsed_response
        
        logger.warning(f"Failed to parse JSON response for state '%s'. Response: %s", state_name, response_text)
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