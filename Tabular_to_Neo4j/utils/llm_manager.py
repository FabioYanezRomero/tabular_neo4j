"""High-level helpers for invoking language models."""

from typing import Dict, Any

from Tabular_to_Neo4j.config import LLM_CONFIGS
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


def call_llm_with_state(state_name: str, prompt: str) -> str:
    """Call the LLM for a specific pipeline state."""
    try:
        template_name = f"{state_name}_prompt.txt"
        save_prompt_sample(template_name, prompt, {"state_name": state_name})
        logger.debug("Saved prompt sample for state '%s'", state_name)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(
            "Failed to save prompt sample for state '%s': %s", state_name, exc
        )

    with load_llm_for_state(state_name) as llm_func:
        response = llm_func(prompt)
        try:
            save_prompt_sample(
                f"{state_name}_response.txt",
                response,
                {"state_name": state_name},
                is_template=False,
            )
            logger.debug("Saved response sample for state '%s'", state_name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "Failed to save response sample for state '%s': %s", state_name, exc
            )
        return response


def call_llm_with_json_output(
    prompt: str,
    state_name: str = None,
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
        state_name = next(iter(LLM_CONFIGS.keys()), None)

    try:
        template_name = f"{state_name}_original_prompt.txt"
        save_prompt_sample(template_name, prompt, {"state_name": state_name})
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(
            "Failed to save original prompt sample for state '%s': %s", state_name, exc
        )

    if LMSTUDIO_AVAILABLE:
        try:
            json_prompt = (
                f"{prompt}\n\nPlease provide your response in valid JSON format."
            )
            try:
                template_name = f"{state_name}_json_prompt.txt"
                save_prompt_sample(
                    template_name, json_prompt, {"state_name": state_name}
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "Failed to save JSON prompt sample for state '%s': %s",
                    state_name,
                    exc,
                )
            from Tabular_to_Neo4j.utils.lmstudio_client import get_lmstudio_client

            client = get_lmstudio_client()
            logger.debug("Calling LMStudio API for state '%s'", state_name)
            response = client.completion(json_prompt)
            response_text = client.extract_completion_text(response)
            try:
                save_prompt_sample(
                    f"{state_name}_json_response.txt",
                    response_text,
                    {"state_name": state_name},
                    is_template=False,
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "Failed to save JSON response sample for state '%s': %s",
                    state_name,
                    exc,
                )
            json_data = extract_json_from_llm_response(response_text)
            if json_data:
                return json_data
            logger.warning(
                "Failed to parse JSON response from LMStudio for state '%s'. Response: %s",
                state_name,
                response_text,
            )
            retry_prompt = f"{prompt}\n\nYou MUST respond with ONLY valid JSON. No other text. No markdown formatting."
            try:
                template_name = f"{state_name}_retry_prompt.txt"
                save_prompt_sample(
                    template_name, retry_prompt, {"state_name": state_name}
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "Failed to save retry prompt sample for state '%s': %s",
                    state_name,
                    exc,
                )
            retry_response = client.completion(retry_prompt)
            retry_text = client.extract_completion_text(retry_response)
            try:
                save_prompt_sample(
                    f"{state_name}_retry_response.txt",
                    retry_text,
                    {"state_name": state_name},
                    is_template=False,
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "Failed to save retry response sample for state '%s': %s",
                    state_name,
                    exc,
                )
            retry_json = extract_json_from_llm_response(retry_text)
            if retry_json:
                return retry_json
            logger.error(
                "Failed to parse JSON from retry response from LMStudio for state '%s'. Response: %s",
                state_name,
                retry_text,
            )
            return {}
        except Exception as exc:
            logger.error("Error using LMStudio for state '%s': %s", state_name, exc)
            logger.warning("Falling back to default LLM provider")

    json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
    try:
        template_name = f"{state_name}_json_prompt.txt"
        save_prompt_sample(template_name, json_prompt, {"state_name": state_name})
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(
            "Failed to save JSON prompt sample for state '%s': %s", state_name, exc
        )
    response = call_llm_with_state(state_name, json_prompt)
    try:
        json_data = extract_json_from_llm_response(response)
        if json_data:
            return json_data
        logger.warning(
            "Failed to parse JSON response for state '%s'. Response: %s",
            state_name,
            response,
        )
        retry_prompt = f"{prompt}\n\nYou MUST respond with ONLY valid JSON. No other text. No markdown formatting."
        try:
            template_name = f"{state_name}_retry_prompt.txt"
            save_prompt_sample(template_name, retry_prompt, {"state_name": state_name})
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "Failed to save retry prompt sample for state '%s': %s", state_name, exc
            )
        retry_response = call_llm_with_state(state_name, retry_prompt)
        retry_json = extract_json_from_llm_response(retry_response)
        if retry_json:
            return retry_json
        logger.error(
            "Failed to parse JSON from retry response for state '%s'. Response: %s",
            state_name,
            retry_response,
        )
        return {}
    except Exception as exc:
        logger.error("Error parsing JSON response for state '%s': %s", state_name, exc)
        return {"error": str(exc), "raw_response": response}
