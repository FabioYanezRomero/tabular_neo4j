"""
LLM Manager for loading and unloading GGUF models via LMStudio for different states.
This module handles per-state LLM configuration, loading, and unloading.
"""

import os
import json
import time
import random
import gc
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import requests
from pathlib import Path
import logging
import re
from contextlib import contextmanager

# Import configuration
from Tabular_to_Neo4j.config import (
    LLM_API_KEY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    LLM_CONFIGS,
    LMSTUDIO_BASE_URL,
    OPENAI_MODEL_NAME
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to terminal
    ]
)
logger = logging.getLogger(__name__)

# Global tracking of loaded models in LMStudio
# This will store information about which models are currently loaded in LMStudio
_LOADED_MODELS = {}

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value to use
    """
    random.seed(seed)
    logger.info(f"Set random seed to {seed} for reproducibility")

def extract_json_from_llm_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from an LLM response, handling various formats.
    
    Args:
        response: The raw LLM response text
        
    Returns:
        Parsed JSON as a dictionary
    """
    # Try to find JSON within triple backticks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON within curly braces
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Just use the whole response
            json_str = response
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        logger.debug(f"Response was: {response}")
        return {"error": "Failed to parse JSON", "raw_response": response}

def call_openai_api(prompt: str, temperature: float = 0.0, model: str = OPENAI_MODEL_NAME, max_retries: int = 3) -> str:
    """
    Call the OpenAI API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
        temperature: Temperature setting (0.0 for deterministic results)
        model: The model to use
        max_retries: Maximum number of retries on failure
        
    Returns:
        The LLM response as a string
    """
    import openai
    
    # Use API key from environment if not in config
    api_key = LLM_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set it in config.py or as an environment variable.")
    
    openai.api_key = api_key
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                seed=DEFAULT_SEED  # Use consistent seed for reproducibility
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"OpenAI API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                raise

def get_lmstudio_models() -> List[Dict[str, Any]]:
    """
    Get a list of available models in LMStudio.
    
    Returns:
        List of model information dictionaries
    """
    base_url = LMSTUDIO_BASE_URL
    
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        logger.error(f"Failed to get LMStudio models: {e}")
        return []

def load_model_in_lmstudio(model_name: str, state_name: str = None, max_retries: int = 3) -> bool:
    """
    Load a GGUF model in LMStudio.
    
    Args:
        model_name: Name of the model to load
        state_name: Name of the state requesting the model (for logging)
        max_retries: Maximum number of retries on failure
        
    Returns:
        True if model was loaded successfully, False otherwise
    """
    base_url = LMSTUDIO_BASE_URL
    state_info = f" for state '{state_name}'" if state_name else ""
    
    # Check if model is already loaded
    if model_name in _LOADED_MODELS and _LOADED_MODELS[model_name].get("loaded", False):
        logger.info(f"🔄 Model '{model_name}' is already loaded in LMStudio{state_info}")
        return True
    
    logger.info(f"🔍 Searching for model '{model_name}' in LMStudio...")
    
    # Get available models
    models = get_lmstudio_models()
    model_id = None
    
    # Find the model ID by name
    for model in models:
        if model.get("name") == model_name:
            model_id = model.get("id")
            break
    
    if not model_id:
        logger.error(f"❌ Model '{model_name}' not found in LMStudio{state_info}")
        return False
    
    # Load the model
    logger.info(f"⏳ Loading model '{model_name}'{state_info}...")
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/models/{model_id}/load",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
            # Track the loaded model
            _LOADED_MODELS[model_name] = {
                "id": model_id,
                "loaded": True,
                "loaded_at": time.time(),
                "last_state": state_name
            }
            
            load_time = time.time() - start_time
            logger.info(f"✅ Successfully loaded model '{model_name}'{state_info} in {load_time:.2f} seconds")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"⚠️ Failed to load model '{model_name}'{state_info}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to load model '{model_name}'{state_info} after {max_retries} attempts: {e}")
                return False

def unload_model_from_lmstudio(model_name: str, state_name: str = None, max_retries: int = 3) -> bool:
    """
    Unload a GGUF model from LMStudio.
    
    Args:
        model_name: Name of the model to unload
        state_name: Name of the state that was using the model (for logging)
        max_retries: Maximum number of retries on failure
        
    Returns:
        True if model was unloaded successfully, False otherwise
    """
    base_url = LMSTUDIO_BASE_URL
    state_info = f" for state '{state_name}'" if state_name else ""
    
    # Check if model is loaded
    if model_name not in _LOADED_MODELS or not _LOADED_MODELS[model_name].get("loaded", False):
        logger.info(f"ℹ️ Model '{model_name}' is not loaded in LMStudio{state_info}")
        return True
    
    model_id = _LOADED_MODELS[model_name].get("id")
    if not model_id:
        logger.error(f"❌ Model ID for '{model_name}' not found{state_info}")
        return False
    
    # Calculate how long the model was loaded
    loaded_at = _LOADED_MODELS[model_name].get("loaded_at", time.time())
    loaded_duration = time.time() - loaded_at
    
    # Unload the model
    logger.info(f"⏳ Unloading model '{model_name}'{state_info}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/models/{model_id}/unload",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
            # Update tracking
            if model_name in _LOADED_MODELS:
                _LOADED_MODELS[model_name]["loaded"] = False
                _LOADED_MODELS[model_name]["unloaded_at"] = time.time()
            
            logger.info(f"✅ Successfully unloaded model '{model_name}'{state_info} (was loaded for {loaded_duration:.2f} seconds)")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"⚠️ Failed to unload model '{model_name}'{state_info}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to unload model '{model_name}'{state_info} after {max_retries} attempts: {e}")
                return False

def call_lmstudio_api(prompt: str, model_name: str = None, temperature: float = 0.0, seed: int = DEFAULT_SEED, state_name: str = None, max_retries: int = 3) -> str:
    """
    Call the LMStudio API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
        model_name: Name of the model to use (must be loaded in LMStudio)
        temperature: Temperature setting (0.0 for deterministic results)
        seed: Seed for reproducibility
        state_name: Name of the state using the model (for logging)
        max_retries: Maximum number of retries on failure
        
    Returns:
        The LLM response as a string
    """
    base_url = LMSTUDIO_BASE_URL
    state_info = f" for state '{state_name}'" if state_name else ""
    
    # Ensure the model is loaded
    if model_name and model_name not in _LOADED_MODELS:
        logger.warning(f"⚠️ Model '{model_name}' is not tracked as loaded{state_info}. Attempting to load it now.")
        if not load_model_in_lmstudio(model_name, state_name):
            raise ValueError(f"Failed to load model '{model_name}' in LMStudio{state_info}")
    
    # Log the API call
    truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
    logger.info(f"💬 Sending prompt to model '{model_name}'{state_info} (temp={temperature}, seed={seed})")
    
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "seed": seed
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            # Log success
            response_time = time.time() - start_time
            truncated_response = content[:50] + "..." if len(content) > 50 else content
            logger.info(f"💬 Received response from model '{model_name}'{state_info} in {response_time:.2f} seconds")
            
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"⚠️ LMStudio API call failed{state_info}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ LMStudio API call failed{state_info} after {max_retries} attempts: {e}")
                raise

def call_huggingface_api(prompt: str, model_id: str, temperature: float = 0.0, seed: int = DEFAULT_SEED, max_retries: int = 3) -> str:
    """
    Call the Hugging Face API with retry logic (fallback option).
    
    Args:
        prompt: The prompt to send to the API
        model_id: The model ID to use
        temperature: Temperature setting (0.0 for deterministic results)
        seed: Seed for reproducibility
        max_retries: Maximum number of retries on failure
        
    Returns:
        The LLM response as a string
    """
    # Use API key from environment if not in config
    api_key = LLM_API_KEY or os.environ.get("HF_API_KEY")
    if not api_key:
        raise ValueError("Hugging Face API key not found. Set it in config.py or as an environment variable.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}",
                headers=headers,
                json={
                    "inputs": prompt, 
                    "parameters": {
                        "temperature": temperature,
                        "seed": seed,
                        "return_full_text": False
                    }
                }
            )
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Hugging Face API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Hugging Face API call failed after {max_retries} attempts: {e}")
                raise

@contextmanager
def load_llm_for_state(state_name: str):
    """
    Context manager to load and unload an LLM for a specific state.
    
    Args:
        state_name: The name of the state to load the LLM for
        
    Yields:
        A function to call the loaded LLM
    """
    # Get the LLM configuration for this state
    config = LLM_CONFIGS.get(state_name, {})
    provider = config.get("provider", DEFAULT_LLM_PROVIDER)
    model_name = config.get("model_name", None)
    temperature = config.get("temperature", DEFAULT_TEMPERATURE)
    seed = config.get("seed", DEFAULT_SEED)
    auto_load = config.get("auto_load", True)
    auto_unload = config.get("auto_unload", True)
    output_format = config.get("output_format", {})
    
    logger.info(f"Setting up LLM for state '{state_name}' using provider '{provider}' and model '{model_name}'")
    
    # Load the model if auto_load is enabled
    if auto_load and provider == "lmstudio" and model_name:
        load_model_in_lmstudio(model_name, state_name)
    
    # Define a function to call the LLM
    def call_llm_func(prompt: str) -> str:
        nonlocal provider, model_name, temperature, seed, output_format, state_name
        
        # Add format instructions to the prompt if specified
        if output_format:
            format_type = output_format.get("type", "")
            format_example = output_format.get("example", "")
            
            if format_type and format_example:
                format_instruction = f"\n\nPlease provide your response in {format_type} format. Example: {format_example}"
                prompt += format_instruction
        
        # Set seed for reproducibility
        set_seed(seed)
        
        if provider == "openai":
            return call_openai_api(prompt, temperature)
        elif provider == "lmstudio":
            return call_lmstudio_api(prompt, model_name, temperature, seed, state_name)
        elif provider == "huggingface":
            # Fallback to Hugging Face API
            return call_huggingface_api(prompt, model_name, temperature, seed)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    try:
        # Yield the function to call the LLM
        yield call_llm_func
    finally:
        # Unload the model if auto_unload is enabled
        if auto_unload and provider == "lmstudio" and model_name:
            unload_model_from_lmstudio(model_name, state_name)

def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        template_name: Name of the template file (e.g., 'infer_header.txt')
        
    Returns:
        The prompt template as a string
    """
    base_dir = Path(__file__).parent.parent
    prompt_path = base_dir / "prompts" / template_name
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error loading prompt template {template_name}: {e}")
        # Return a basic template if the file doesn't exist
        return "Please analyze the following data: {data}"

def format_prompt(template_name: str, **kwargs) -> str:
    """
    Format a prompt template with the given arguments.
    
    Args:
        template_name: Name of the template file
        **kwargs: Arguments to format the template with
        
    Returns:
        The formatted prompt as a string
    """
    template = load_prompt_template(template_name)
    return template.format(**kwargs)

def call_llm_with_state(state_name: str, prompt: str) -> str:
    """
    Call the LLM for a specific state.
    
    Args:
        state_name: The name of the state to use the LLM for
        prompt: The prompt to send to the LLM
        
    Returns:
        The LLM response as a string
    """
    with load_llm_for_state(state_name) as llm_func:
        return llm_func(prompt)

def call_llm_with_json_output(prompt: str, state_name: str = None, is_translation: bool = False) -> Dict[str, Any]:
    """
    Call the LLM and parse the response as JSON.
    
    Args:
        prompt: The prompt to send to the LLM
        state_name: The name of the state to use the LLM for
        is_translation: Whether this is a translation task
        
    Returns:
        The parsed JSON response as a dictionary
    """
    # If translation is requested, use the translate_header state
    if is_translation and state_name is None:
        state_name = "translate_header"
    
    # If no state is specified, use a default state
    if state_name is None:
        state_name = next(iter(LLM_CONFIGS.keys()), None)
    
    # Add explicit JSON formatting instruction
    json_instruction = "\n\nPlease respond with valid JSON only."
    full_prompt = prompt + json_instruction
    
    # Call the LLM for the specified state
    response = call_llm_with_state(state_name, full_prompt)
    
    # Parse the response as JSON
    return extract_json_from_llm_response(response)

def get_model_info(state_name: str) -> Dict[str, Any]:
    """
    Get information about the model configured for a specific state.
    
    Args:
        state_name: The name of the state
        
    Returns:
        Dictionary with model information
    """
    config = LLM_CONFIGS.get(state_name, {})
    provider = config.get("provider", DEFAULT_LLM_PROVIDER)
    model_name = config.get("model_name", "Unknown")
    description = config.get("description", "")
    output_format = config.get("output_format", {})
    auto_load = config.get("auto_load", True)
    auto_unload = config.get("auto_unload", True)
    
    # Check if the model is currently loaded
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
        "output_format": output_format
    }


def list_loaded_models() -> List[Dict[str, Any]]:
    """
    List all models that are currently loaded or have been loaded.
    
    Returns:
        List of dictionaries with model information
    """
    models = []
    
    for model_name, info in _LOADED_MODELS.items():
        models.append({
            "name": model_name,
            "id": info.get("id"),
            "is_loaded": info.get("loaded", False),
            "loaded_at": info.get("loaded_at"),
            "unloaded_at": info.get("unloaded_at", None),
            "last_state": info.get("last_state")
        })
    
    return models
