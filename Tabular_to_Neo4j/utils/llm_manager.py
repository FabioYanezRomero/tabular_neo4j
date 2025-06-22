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
    """Call the LLM for a specific pipeline state."""
    try:
        template_name = f"{state_name}_prompt.txt"
        save_prompt_sample(template_name, prompt, {"state_name": state_name})
        logger.debug("Saved prompt sample for state '%s'", state_name)
    except Exception as exc:
        logger.warning("Failed to save prompt sample for state '%s': %s", state_name, exc)

    # Dynamic provider selection
    provider = "lm_studio"
    if config and "configurable" in config:
        provider = config["configurable"].get("llm_provider", "lm_studio")
    if provider == "lm_studio":
        model = models["lm_studio"]()
    elif provider == "ollama":
        model_name = None
        if config and "configurable" in config:
            model_name = config["configurable"].get("llm_model")
        model = models["ollama"](model_name)
    else:
        logger.error(f"LLM provider '{provider}' not found. Please check your configuration.")
        raise ValueError(f"LLM provider '{provider}' not found in llm_providers.models. Valid options are: {list(models.keys())}")
    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, "content") else str(response)
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
        logger.error(f"Error invoking LLM provider '{provider}': {exc}")
        return ""


def call_llm_with_json_output(
    prompt: str,
    state_name: str = None,
    config: dict = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call the LLM and parse the response as JSON.
    
    Args:
        prompt: The prompt to send to the LLM
        state_name: The name of the state to use the LLM for
        **kwargs: Optional parameters for future extensions
        
    Returns:
        The parsed JSON response as a dictionary
    """
    # If no state is specified, use a default state

    if state_name is None:
        state_name = "default"

    try:
        template_name = f"{state_name}_original_prompt.txt"
        save_prompt_sample(template_name, prompt, {"state_name": state_name})
    except Exception as exc:
        logger.warning("Failed to save original prompt sample for state '%s': %s", state_name, exc)

    json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
    provider = "lm_studio"
    if config and "configurable" in config:
        provider = config["configurable"].get("llm_provider", "lm_studio")
    if provider == "lm_studio":
        model = models["lm_studio"]()
    elif provider == "ollama":
        model_name = None
        if config and "configurable" in config:
            model_name = config["configurable"].get("llm_model")
        model = models["ollama"](model_name)
    else:
        logger.error(f"LLM provider '{provider}' not found. Please check your configuration.")
        raise ValueError(f"LLM provider '{provider}' not found in llm_providers.models. Valid options are: {list(models.keys())}")
    try:
        response = model.invoke([{"role": "user", "content": json_prompt}])
        response_text = response.content if hasattr(response, "content") else str(response)
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
        retry_response = model.invoke([{"role": "user", "content": retry_prompt}])
        retry_text = retry_response.content if hasattr(retry_response, "content") else str(retry_response)
        try:
            template_name = f"{state_name}_retry_response.txt"
            save_prompt_sample(
                template_name,
                retry_text,
                {"state_name": state_name},
                is_template=False,
            )
        except Exception as exc:
            logger.warning("Failed to save retry response sample for state '%s': %s", state_name, exc)
        retry_json = extract_json_from_llm_response(retry_text)
        if retry_json:
            return retry_json
        logger.error(f"Failed to parse JSON from retry response for state '%s'. Response: %s", state_name, retry_text)
        return {}
    except Exception as exc:
        logger.error(f"Error invoking LLM provider '{provider}': {exc}")
        return {"error": str(exc), "raw_response": ""}
