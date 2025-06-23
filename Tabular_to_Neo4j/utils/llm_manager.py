"""High-level helpers for invoking language models."""

from typing import Dict, Any

from Tabular_to_Neo4j.config.llm_providers import models
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.utils.llm_api import (
    LMSTUDIO_AVAILABLE,
    load_llm_for_state,
    get_model_info,
    list_loaded_models,
)
from Tabular_to_Neo4j.utils.prompt_utils import (
    save_prompt_sample,
    format_prompt,
)
from Tabular_to_Neo4j.utils.response_utils import extract_json_from_llm_response

logger = get_logger(__name__)

__all__ = [
    "format_prompt",
    "call_llm_with_state",
    "call_llm_with_json_output",
    "reset_prompt_sample_directory",
    "save_prompt_sample",
    "get_model_info",
    "list_loaded_models",
]

from Tabular_to_Neo4j.utils.prompt_utils import (
    reset_prompt_sample_directory,
)  # noqa: E402


def call_llm_with_state(state_name: str, prompt: str, config: dict = None) -> str:
    """
    Call the LLM for a specific pipeline state using the unified dispatcher.
    Logs prompt and response for debugging.
    """
    from Tabular_to_Neo4j.config.settings import LLM_CONFIGS
    from Tabular_to_Neo4j.utils.llm_api import call_llm_api

    # Save prompt sample for traceability
    try:
        template_name = f"{state_name}_prompt.txt"
        save_prompt_sample(template_name, prompt, {"state_name": state_name})
        logger.debug("Saved prompt sample for state '%s'", state_name)
    except Exception as exc:
        logger.warning("Failed to save prompt sample for state '%s': %s", state_name, exc)

    # Use LLM_CONFIGS for state-specific config
    state_config = LLM_CONFIGS.get(state_name, {})
    if config:
        state_config.update(config)

    try:
        # Unified call via llm_api dispatcher
        response_text = call_llm_api(prompt, state_config)
        logger.info(f"[LLM][{state_name}] Prompt: {prompt}")
        logger.info(f"[LLM][{state_name}] Response: {response_text}")
        try:
            save_prompt_sample(
                f"{state_name}_response.txt",
                response_text,
                {"state_name": state_name},
                is_template=False,
            )
            logger.debug("Saved response sample for state '%s'", state_name)
        except Exception as exc:
            logger.warning("Failed to save response sample for state '%s': %s", state_name, exc)
        return response_text
    except Exception as exc:
        logger.error(f"Error invoking LLM for state '{state_name}': {exc}")
        return ""


def call_llm_with_json_output(
    prompt: str,
    state_name: str = None,
    config: dict = None,
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
        template_name = f"{state_name}_original_prompt.txt"
        save_prompt_sample(template_name, prompt, {"state_name": state_name})
    except Exception as exc:
        logger.warning("Failed to save original prompt sample for state '%s': %s", state_name, exc)

    # Augment prompt to ask for JSON output
    json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
    state_config = LLM_CONFIGS.get(state_name, {})
    if config:
        state_config.update(config)

    try:
        response_text = call_llm_api(json_prompt, state_config)
        logger.info(f"[LLM][{state_name}] Prompt: {json_prompt}")
        logger.info(f"[LLM][{state_name}] Response: {response_text}")
        try:
            template_name = f"{state_name}_json_response.txt"
            save_prompt_sample(
                template_name,
                response_text,
                {"state_name": state_name},
                is_template=False,
            )
        except Exception as exc:
            logger.warning("Failed to save JSON response sample for state '%s': %s", state_name, exc)
        json_data = extract_json_from_llm_response(response_text)
        if json_data:
            return json_data
        logger.warning(f"Failed to parse JSON response for state '%s'. Response: %s", state_name, response_text)
        # Retry with stricter prompt
        retry_prompt = f"{prompt}\n\nYou MUST respond with ONLY valid JSON. No other text. No markdown formatting."
        try:
            template_name = f"{state_name}_retry_prompt.txt"
            save_prompt_sample(template_name, retry_prompt, {"state_name": state_name})
        except Exception as exc:
            logger.warning("Failed to save retry prompt sample for state '%s': %s", state_name, exc)
        retry_response_text = call_llm_api(retry_prompt, state_config)
        try:
            template_name = f"{state_name}_retry_response.txt"
            save_prompt_sample(
                template_name,
                retry_response_text,
                {"state_name": state_name},
                is_template=False,
            )
        except Exception as exc:
            logger.warning("Failed to save retry response sample for state '%s': %s", state_name, exc)
        retry_json = extract_json_from_llm_response(retry_response_text)
        if retry_json:
            return retry_json
        logger.error(f"Failed to parse JSON from retry response for state '%s'. Response: %s", state_name, retry_response_text)
        return {}
    except Exception as exc:
        logger.error(f"Error invoking LLM for state '{state_name}': {exc}")
        return {"error": str(exc), "raw_response": ""}
