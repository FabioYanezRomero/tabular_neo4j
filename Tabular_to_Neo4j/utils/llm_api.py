"""API helpers for interacting with LMStudio models."""

import os
import logging
import random
import time
import requests
import json
from contextlib import contextmanager
from typing import Any, Dict, List

from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.config.settings import DEFAULT_SEED, DEFAULT_TEMPERATURE, LLM_CONFIGS, DEFAULT_LLM_PROVIDER
from Tabular_to_Neo4j.utils.ollama_api import load_model_in_ollama, unload_model_from_ollama

LMSTUDIO_AVAILABLE = False
OLLAMA_AVAILABLE = False

# Try to import LMStudio config only if needed
if DEFAULT_LLM_PROVIDER == "lmstudio":
    try:
        from Tabular_to_Neo4j.config.lmstudio_config import (
            LMSTUDIO_ENDPOINT,
            LMSTUDIO_BASE_URL,
            DEFAULT_MODEL,
        )
        from Tabular_to_Neo4j.utils.lmstudio_client import get_lmstudio_client
        LMSTUDIO_AVAILABLE = True
    except Exception:
        logger = get_logger(__name__)
        logger.warning("LMStudio configuration not found, LMStudio features disabled.")

# Ollama config always importable
try:
    from Tabular_to_Neo4j.utils.ollama_api import load_model_in_ollama, unload_model_from_ollama
    OLLAMA_AVAILABLE = True
except Exception:
    logger = get_logger(__name__)
    logger.warning("Ollama API not available, Ollama features disabled.")
    OLLAMA_AVAILABLE = False

logger = get_logger(__name__)

_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}


def _is_model_loaded(model_name: str) -> bool:
    if model_name in _LOADED_MODELS:
        return True
    try:
        models = get_lmstudio_models()
        for model in models:
            if model.get("id") == model_name and model.get("status") == "loaded":
                _LOADED_MODELS[model_name] = {
                    "loaded_at": time.time(),
                    "loaded_by": "external",
                }
                return True
        return False
    except Exception as e:
        logger.warning(f"Error checking if model '{model_name}' is loaded: {e}")
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    logger.info(f"Set random seed to {seed} for reproducibility")


def get_lmstudio_models() -> List[Dict[str, Any]]:
    # Normalize base_url: remove trailing slash and trailing '/v1' if present
    base_url = LMSTUDIO_BASE_URL.rstrip('/')
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    try:
        response = requests.get(
            f"{base_url}/v1/models", headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        logger.error(f"Failed to get LMStudio models: {e}")
        return []


def load_model_in_lmstudio(
    model_name: str, state_name: str | None = None, max_retries: int = 3
) -> bool:
    base_url = LMSTUDIO_BASE_URL
    state_info = f" for state '{state_name}'" if state_name else ""

    if model_name in _LOADED_MODELS and _LOADED_MODELS[model_name].get("loaded", False):
        logger.info(
            f"🔄 Model '{model_name}' is already loaded in LMStudio{state_info}"
        )
        return True

    logger.info(f"🔍 Searching for model '{model_name}' in LMStudio...")
    models = get_lmstudio_models()
    model_id = None
    for model in models:
        # Accept match if either 'id' or 'name' matches model_name
        if model.get("id") == model_name or model.get("name") == model_name:
            model_id = model.get("id")
            break
    if not model_id:
        logger.error(f"❌ Model '{model_name}' not found in LMStudio{state_info} (available: {[m.get('id') for m in models]})")
        return False

    logger.info(f"⏳ Loading model '{model_name}'{state_info}...")
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/models/{model_id}/load",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            _LOADED_MODELS[model_name] = {
                "id": model_id,
                "loaded": True,
                "loaded_at": time.time(),
                "last_state": state_name,
            }
            load_time = time.time() - start_time
            logger.info(
                f"✅ Successfully loaded model '{model_name}'{state_info} in {load_time:.2f} seconds"
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"⚠️ Failed to load model '{model_name}'{state_info}: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"❌ Failed to load model '{model_name}'{state_info} after {max_retries} attempts: {e}"
                )
                return False


def unload_model_from_lmstudio(
    model_name: str, state_name: str | None = None, max_retries: int = 3
) -> bool:
    base_url = LMSTUDIO_BASE_URL
    state_info = f" for state '{state_name}'" if state_name else ""

    if model_name not in _LOADED_MODELS or not _LOADED_MODELS[model_name].get(
        "loaded", False
    ):
        logger.info(f"ℹ️ Model '{model_name}' is not loaded in LMStudio{state_info}")
        return True

    model_id = _LOADED_MODELS[model_name].get("id")
    if not model_id:
        logger.error(f"❌ Model ID for '{model_name}' not found{state_info}")
        return False

    loaded_at = _LOADED_MODELS[model_name].get("loaded_at", time.time())
    loaded_duration = time.time() - loaded_at
    logger.info(f"⏳ Unloading model '{model_name}'{state_info}...")

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/models/{model_id}/unload",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            if model_name in _LOADED_MODELS:
                _LOADED_MODELS[model_name]["loaded"] = False
                _LOADED_MODELS[model_name]["unloaded_at"] = time.time()
            logger.info(
                f"✅ Successfully unloaded model '{model_name}'{state_info} (was loaded for {loaded_duration:.2f} seconds)"
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"⚠️ Failed to unload model '{model_name}'{state_info}: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"❌ Failed to unload model '{model_name}'{state_info} after {max_retries} attempts: {e}"
                )
                return False


def call_ollama_api(
    prompt: str,
    model_name: str = None,
    temperature: float = 0.7,
    seed: int = DEFAULT_SEED,
    state_name: str = None,
    max_retries: int = 3,
    url: str = None,
) -> str:
    """
    Call Ollama API for chat completion.
    """
    set_seed(seed)
    if not model_name:
        raise ValueError("Model name is required for Ollama API call")
    if not url:
        raise ValueError("URL is required for Ollama API call")
    url = f"{url}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": False,
        "num_predict": 2048,
    }
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            response_data = response.json()
            if "message" in response_data:
                return response_data["message"]["content"]
            elif "choices" in response_data and len(response_data["choices"]):
                return response_data["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected Ollama API response: {response_data}")
                return str(response_data)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Ollama API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Ollama API call failed after {max_retries} attempts: {e}")
                raise

def call_lmstudio_api(
    prompt: str,
    model_name: str | None = None,
    temperature: float = 0.7,
    seed: int = DEFAULT_SEED,
    state_name: str | None = None,
    max_retries: int = 3,
) -> str:
    from Tabular_to_Neo4j.config.settings import LLM_CONFIGS, DEFAULT_LMSTUDIO_MODEL
    set_seed(seed)
    base_url = LMSTUDIO_BASE_URL
    if not model_name:
        model_name = DEFAULT_LMSTUDIO_MODEL
        logger.info(f"No model specified, using default model: {model_name}")
    system_prompt = (
        "You are a helpful assistant that specializes in data analysis and Neo4j graph modeling. "
        "Always return valid JSON when asked to do so."
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}
    logger.debug(
        f"Calling LMStudio API with model '{model_name}', temperature {temperature}"
    )
    logger.debug(f"Prompt length: {len(prompt)} characters")
    for attempt in range(max_retries):
        try:
            logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
            try:
                response_data = response.json()
                logger.debug(
                    f"Received response from LMStudio API: {str(response_data)[:200]}..."
                )
                if "choices" in response_data and len(response_data["choices"]):
                    if "message" in response_data["choices"][0]:
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Unexpected response format: {response_data}")
                        return str(response_data)
                else:
                    logger.error(f"No choices in response: {response_data}")
                    return str(response_data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response.text[:500]}...")
                return response.text
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Connection error: Could not connect to LM Studio at {base_url}"
            )
            logger.error(
                "Please ensure LM Studio is running and accessible at the configured URL"
            )
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise ValueError(
                    f"Could not connect to LM Studio after {max_retries} attempts. Please ensure LM Studio is running at {base_url}"
                )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"LMStudio API call failed: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"LMStudio API call failed after {max_retries} attempts: {e}"
                )
                raise
    raise RuntimeError(
        f"Failed to get a valid response from LMStudio after {max_retries} attempts"
    )


def call_llm_api(prompt: str, config: dict = None) -> str:
    """
    Unified LLM dispatcher for all providers (Ollama, LMStudio).
    Args:
        prompt: The prompt to send to the LLM.
        config: Dictionary with provider/model/temperature/seed, etc.
    Returns:
        The LLM response as a string.
    """
    if config is None:
        config = {}
    from Tabular_to_Neo4j.config.settings import OLLAMA_URL, DEFAULT_LLM_PROVIDER
    provider = config.get("provider", DEFAULT_LLM_PROVIDER)
    model_name = config.get("model_name") or config.get("model")
    temperature = config.get("temperature", DEFAULT_TEMPERATURE)
    seed = config.get("seed", DEFAULT_SEED)
    state_name = config.get("state_name") if "state_name" in config else None
    if provider == "ollama":
        url = config.get("url") or OLLAMA_URL
        return call_ollama_api(prompt, model_name, temperature, seed, state_name, url=url)
    elif provider == "lmstudio":
        return call_lmstudio_api(prompt, model_name, temperature, seed, state_name)
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

    from Tabular_to_Neo4j.config.settings import LLM_CONFIGS, DEFAULT_LMSTUDIO_MODEL

    set_seed(seed)
    base_url = LMSTUDIO_BASE_URL
    if not model_name:
        model_name = DEFAULT_LMSTUDIO_MODEL
        logger.info(f"No model specified, using default model: {model_name}")

    system_prompt = (
        "You are a helpful assistant that specializes in data analysis and Neo4j graph modeling. "
        "Always return valid JSON when asked to do so."
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}

    logger.debug(
        f"Calling LMStudio API with model '{model_name}', temperature {temperature}"
    )
    logger.debug(f"Prompt length: {len(prompt)} characters")

    for attempt in range(max_retries):
        try:
            logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
            try:
                response_data = response.json()
                logger.debug(
                    f"Received response from LMStudio API: {str(response_data)[:200]}..."
                )
                if "choices" in response_data and len(response_data["choices"]):
                    if "message" in response_data["choices"][0]:
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Unexpected response format: {response_data}")
                        return str(response_data)
                else:
                    logger.error(f"No choices in response: {response_data}")
                    return str(response_data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response.text[:500]}...")
                return response.text
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Connection error: Could not connect to LM Studio at {base_url}"
            )
            logger.error(
                "Please ensure LM Studio is running and accessible at the configured URL"
            )
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise ValueError(
                    f"Could not connect to LM Studio after {max_retries} attempts. Please ensure LM Studio is running at {base_url}"
                )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"LMStudio API call failed: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"LMStudio API call failed after {max_retries} attempts: {e}"
                )
                raise
    raise RuntimeError(
        f"Failed to get a valid response from LMStudio after {max_retries} attempts"
    )


@contextmanager
def load_llm_for_state(state_name: str):
    config = LLM_CONFIGS.get(state_name, {})
    provider = config.get("provider", "lmstudio")
    model_name = config.get("model_name") or config.get("model")
    if not model_name:
        logger.warning(f"No model_name or model specified in LLM_CONFIGS for state '{state_name}', will fall back to default model.")
    temperature = config.get("temperature", DEFAULT_TEMPERATURE)
    seed = config.get("seed", DEFAULT_SEED)
    auto_load = config.get("auto_load", True)
    auto_unload = config.get("auto_unload", True)
    output_format = config.get("output_format", {})

    logger.info(
        f"Setting up LLM for state '{state_name}' using provider '{provider}' and model '{model_name}'"
    )

    # Load model if needed
    if auto_load and model_name:
        if provider == "lmstudio":
            load_model_in_lmstudio(model_name, state_name)
        elif provider == "ollama":
            load_model_in_ollama(model_name)

    def call_llm_func(prompt: str) -> str:
        nonlocal provider, model_name, temperature, seed, output_format, state_name
        if output_format:
            format_type = output_format.get("type", "")
            format_example = output_format.get("example", "")
            if format_type and format_example:
                format_instruction = f"\n\nPlease provide your response in {format_type} format. Example: {format_example}"
                prompt += format_instruction
        set_seed(seed)
        if provider == "lmstudio" and LMSTUDIO_AVAILABLE:
            return call_lmstudio_api(prompt, model_name, temperature, seed, state_name)
        elif provider == "ollama" and OLLAMA_AVAILABLE:
            from Tabular_to_Neo4j.config.settings import OLLAMA_URL
            return call_ollama_api(prompt, model_name, temperature, seed, state_name, url=OLLAMA_URL)
        else:
            logger.warning(f"Provider '{provider}' is not supported or not available. Using Ollama if possible.")
            if OLLAMA_AVAILABLE:
                from Tabular_to_Neo4j.config.settings import OLLAMA_URL
                return call_ollama_api(prompt, model_name, temperature, seed, state_name, url=OLLAMA_URL)
            elif LMSTUDIO_AVAILABLE:
                return call_lmstudio_api(prompt, model_name, temperature, seed, state_name)
            else:
                raise RuntimeError("No valid LLM provider available.")


    try:
        yield call_llm_func
    finally:
        if auto_unload and model_name:
            if provider == "lmstudio":
                unload_model_from_lmstudio(model_name, state_name)
            elif provider == "ollama":
                unload_model_from_ollama(model_name)



def get_model_info(state_name: str) -> Dict[str, Any]:
    config = LLM_CONFIGS.get(state_name, {})
    provider = config.get("provider", "lmstudio")
    model_name = config.get("model_name", "Unknown")
    description = config.get("description", "")
    output_format = config.get("output_format", {})
    auto_load = config.get("auto_load", True)
    auto_unload = config.get("auto_unload", True)

    is_loaded = False
    loaded_at = None
    if provider == "lmstudio" and model_name in _LOADED_MODELS:
        is_loaded = _LOADED_MODELS[model_name].get("loaded", False)
        loaded_at = _LOADED_MODELS[model_name].get("loaded_at")

    return {
        "state": state_name,
        "provider": provider,
        "model_name": model_name,
        "description": description,
        "is_loaded": is_loaded,
        "loaded_at": loaded_at,
        "auto_load": auto_load,
        "auto_unload": auto_unload,
        "output_format": output_format,
    }


def list_loaded_models() -> List[Dict[str, Any]]:
    models = []
    for model_name, info in _LOADED_MODELS.items():
        models.append(
            {
                "name": model_name,
                "id": info.get("id"),
                "is_loaded": info.get("loaded", False),
                "loaded_at": info.get("loaded_at"),
                "unloaded_at": info.get("unloaded_at", None),
                "last_state": info.get("last_state"),
            }
        )
    return models
