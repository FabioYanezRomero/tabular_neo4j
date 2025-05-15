"""
LLM Manager for loading and unloading GGUF models via LMStudio for different states.
This module handles per-state LLM configuration, loading, and unloading.
"""

import os
import json
import time
import random
import gc
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import requests
from pathlib import Path
from Tabular_to_Neo4j.utils.logging_config import get_logger
import re
from contextlib import contextmanager

# Configure logging
logger = get_logger(__name__)

# Import configuration
from Tabular_to_Neo4j.config import (
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    LLM_CONFIGS
)

# Import LMStudio client
try:
    from Tabular_to_Neo4j.config.lmstudio_config import (
        LMSTUDIO_ENDPOINT,
        LMSTUDIO_BASE_URL,
        DEFAULT_MODEL
    )
    from Tabular_to_Neo4j.utils.lmstudio_client import get_lmstudio_client
    LMSTUDIO_AVAILABLE = True
except ImportError:
    logger.warning("LMStudio configuration not found, falling back to default LLM provider")
    LMSTUDIO_AVAILABLE = False
    # Define a fallback base URL if import fails
    LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# Global tracking of loaded models in LMStudio
# This will store information about which models are currently loaded in LMStudio
_LOADED_MODELS = {}

def _is_model_loaded(model_name: str) -> bool:
    """
    Check if a model is loaded in LMStudio.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if the model is loaded, False otherwise
    """
    # First check our local tracking
    if model_name in _LOADED_MODELS:
        return True
        
    # Then check with LMStudio API
    try:
        models = get_lmstudio_models()
        for model in models:
            if model.get('id') == model_name and model.get('status') == 'loaded':
                # Update our local tracking
                _LOADED_MODELS[model_name] = {
                    'loaded_at': time.time(),
                    'loaded_by': 'external'
                }
                return True
        return False
    except Exception as e:
        logger.warning(f"Error checking if model '{model_name}' is loaded: {e}")
        return False

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
    Extract JSON from an LLM response, handling various formats including LMStudio responses.
    
    Args:
        response: The raw LLM response text
        
    Returns:
        Parsed JSON as a dictionary
    """
    # Handle empty responses
    if not response or response.strip() == "":
        logger.warning("Received empty response from LLM")
        return {"error": "Empty response", "raw_response": ""}
    
    # Log the raw response for debugging
    logger.debug(f"Raw LLM response: {response[:200]}..." if len(response) > 200 else response)
    
    # Clean the response - remove any leading/trailing whitespace and newlines
    cleaned_response = response.strip()
    
    # Try to find JSON within triple backticks (common in markdown formatted responses)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_response)
    if json_match:
        json_str = json_match.group(1).strip()
        logger.debug(f"Found JSON in code block: {json_str[:100]}..." if len(json_str) > 100 else json_str)
    else:
        # Try to find JSON within curly braces (find the outermost complete JSON object)
        # This regex looks for a JSON object that starts with { and ends with }
        json_match = re.search(r'(\{[\s\S]*?\})', cleaned_response)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug(f"Found JSON with braces: {json_str[:100]}..." if len(json_str) > 100 else json_str)
        else:
            # Just use the whole response
            json_str = cleaned_response
            logger.debug(f"Using whole response as JSON: {json_str[:100]}..." if len(json_str) > 100 else json_str)
    
    # If the string doesn't start with {, try to find the first { and parse from there
    if not json_str.startswith("{"):
        brace_index = json_str.find("{")
        if brace_index >= 0:
            json_str = json_str[brace_index:]
            logger.debug(f"Trimmed to start at first brace: {json_str[:100]}..." if len(json_str) > 100 else json_str)
    
    # If the string doesn't end with }, try to find the last } and parse until there
    if not json_str.endswith("}"):
        brace_index = json_str.rfind("}")
        if brace_index >= 0:
            json_str = json_str[:brace_index+1]
            logger.debug(f"Trimmed to end at last brace: {json_str[:100]}..." if len(json_str) > 100 else json_str)
    
    # Special handling for LMStudio responses which might have extra text
    # Sometimes LMStudio adds extra text after the JSON object
    try:
        # First try to parse as is
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If that fails, try to extract just the first valid JSON object
        try:
            # Find all potential JSON objects in the string
            potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str)
            if potential_jsons:
                # Try each potential JSON object until one parses successfully
                for potential_json in potential_jsons:
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error during JSON extraction: {e}")
    
    # If we get here, all parsing attempts failed
    logger.error(f"Failed to parse JSON from LLM response")
    logger.debug(f"Response was: {response}")
    logger.debug(f"Attempted to parse: {json_str}")
    
    # Fallback: try to create a structured response based on content
    if "classification" in response.lower() and "column_name" in response.lower():
        # This is likely an entity classification response
        # Try to extract the classification (entity or property)
        if "entity" in response.lower():
            classification = "entity"
        elif "property" in response.lower():
            classification = "property"
        else:
            classification = "unknown"
            
        # Create a minimal valid response
        return {
            "column_name": re.search(r'column_name[":\s]+(\w+)', response, re.IGNORECASE).group(1) if re.search(r'column_name[":\s]+(\w+)', response, re.IGNORECASE) else "unknown",
            "classification": classification,
            "confidence": 0.5,  # Default confidence
            "reasoning": "Extracted from unstructured response"
        }
    elif "headers" in response.lower():
        # This might be a header validation response
        return {"headers": response}
    else:
        # Generic error response
        return {"error": "Failed to parse JSON", "raw_response": response}

# OpenAI API function removed - using LM Studio exclusively

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

def call_lmstudio_api(prompt: str, model_name: str = None, temperature: float = 0.7, seed: int = DEFAULT_SEED, state_name: str = None, max_retries: int = 3) -> str:
    """
    Call the LMStudio API with retry logic, specifically configured for Gemma model.
    
    Args:
        prompt: The prompt to send to the API
        model_name: Name of the model to use (must be loaded in LMStudio)
        temperature: Temperature setting (0.7 is a good default for Gemma)
        seed: Seed for reproducibility
        state_name: Name of the state using the model (for logging)
        max_retries: Maximum number of retries on failure
        
    Returns:
        The LLM response as a string
    """
    import requests
    import time
    import json
    from Tabular_to_Neo4j.config import DEFAULT_LMSTUDIO_MODEL
    
    # Set the seed for reproducibility
    set_seed(seed)
    
    # Get the base URL for LMStudio
    base_url = LMSTUDIO_BASE_URL
    
    # If no model name is provided, use the default model
    if not model_name:
        model_name = DEFAULT_LMSTUDIO_MODEL
        logger.info(f"No model specified, using default model: {model_name}")
    
    # For Gemma models, we'll use a specific system prompt
    system_prompt = "You are a helpful assistant that specializes in data analysis and Neo4j graph modeling. Always return valid JSON when asked to do so."
    
    # Prepare the request payload for chat completions API
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048,  # Reasonable limit for responses
        "stream": False
    }
    
    # Add seed if provided
    if seed is not None:
        payload["seed"] = seed
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Log the request details for debugging
    logger.debug(f"Calling LMStudio API with model '{model_name}', temperature {temperature}")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    # Make the API call with retries
    for attempt in range(max_retries):
        try:
            logger.debug(f"API call attempt {attempt+1}/{max_retries}")
            
            response = requests.post(
                f"{base_url}/v1/chat/completions",  # Use v1 endpoint for compatibility
                headers=headers,
                data=json.dumps(payload),
                timeout=180  # 3-minute timeout for large responses
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
            
            # Parse the response
            try:
                response_data = response.json()
                logger.debug(f"Received response from LMStudio API: {str(response_data)[:200]}...")
                
                # Extract the content from the response
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    if "message" in response_data["choices"][0]:
                        content = response_data["choices"][0]["message"]["content"]
                        return content
                    else:
                        logger.error(f"Unexpected response format: {response_data}")
                        return str(response_data)  # Return the raw response as a fallback
                else:
                    logger.error(f"No choices in response: {response_data}")
                    return str(response_data)  # Return the raw response as a fallback
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response.text[:500]}...")
                return response.text  # Return the raw text as a fallback
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: Could not connect to LM Studio at {base_url}")
            logger.error("Please ensure LM Studio is running and accessible at the configured URL")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise ValueError(f"Could not connect to LM Studio after {max_retries} attempts. Please ensure LM Studio is running at {base_url}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"LMStudio API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LMStudio API call failed after {max_retries} attempts: {e}")
                raise
    
    # If we get here, all retries failed
    raise RuntimeError(f"Failed to get a valid response from LMStudio after {max_retries} attempts")

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
        
        # Always use LM Studio as the provider
        if provider != "lmstudio":
            logger.warning(f"Provider '{provider}' is not supported. Using LM Studio instead.")
            provider = "lmstudio"
            
        return call_lmstudio_api(prompt, model_name, temperature, seed, state_name)
    
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
    
    # Ensure all kwargs are properly formatted as strings
    formatted_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            formatted_kwargs[key] = str(value)
        elif isinstance(value, list):
            # Convert lists to a readable string format
            formatted_kwargs[key] = str(value)
        elif isinstance(value, dict):
            # Convert dicts to a readable string format
            formatted_kwargs[key] = str(value)
        else:
            formatted_kwargs[key] = str(value) if value is not None else ""
    
    # Always save the template and kwargs for debugging, even if formatting fails
    save_prompt_sample(template_name, template, formatted_kwargs, is_template=True)
    
    try:
        # Use a simple string replacement approach instead of format
        formatted_prompt = template
        for key, value in formatted_kwargs.items():
            placeholder = '{' + key + '}'
            formatted_prompt = formatted_prompt.replace(placeholder, value)
        
        # Save the formatted prompt to the prompt_samples folder
        save_prompt_sample(template_name, formatted_prompt, formatted_kwargs)
        
        return formatted_prompt
    except KeyError as e:
        error_msg = f"Error formatting prompt: missing key {e}. Available keys: {list(formatted_kwargs.keys())}"
        logger.error(f"Missing key in prompt template {template_name}: {e}")
        
        # Save the error message as the formatted prompt for debugging
        save_prompt_sample(template_name, error_msg, formatted_kwargs, is_error=True)
        
        # Return a simplified template with the error message
        return error_msg
    except Exception as e:
        error_msg = f"Error formatting prompt: {str(e)}"
        logger.error(f"Error formatting prompt template {template_name}: {e}")
        
        # Save the error message as the formatted prompt for debugging
        save_prompt_sample(template_name, error_msg, formatted_kwargs, is_error=True)
        
        # Return a simplified template with the error message
        return error_msg


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

# Global variable to store the current run's timestamp directory
_CURRENT_RUN_TIMESTAMP_DIR = None

def reset_prompt_sample_directory():
    """
    Reset the prompt sample directory for a new pipeline run.
    This should be called at the beginning of each pipeline run to ensure
    all prompt samples from the run are stored in a single directory.
    """
    global _CURRENT_RUN_TIMESTAMP_DIR
    _CURRENT_RUN_TIMESTAMP_DIR = None

def save_prompt_sample(template_name: str, formatted_prompt: str, kwargs: dict, is_template: bool = False, is_error: bool = False) -> None:
    """
    Save a formatted prompt sample to the prompt_samples folder.
    All samples from a single run will be stored in the same directory.
    
    Args:
        template_name: Name of the template file
        formatted_prompt: The formatted prompt
        kwargs: The arguments used to format the prompt
        is_template: Whether this is the original template (not formatted)
        is_error: Whether this is an error message
    """
    import os
    import json
    import time
    from pathlib import Path
    
    global _CURRENT_RUN_TIMESTAMP_DIR
    
    base_dir = Path(__file__).parent.parent.parent
    prompt_samples_dir = base_dir / "prompt_samples"
    
    # If this is the first prompt in the run, create a new timestamp directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if _CURRENT_RUN_TIMESTAMP_DIR is None:
        # Create a timestamp-based folder name
        _CURRENT_RUN_TIMESTAMP_DIR = prompt_samples_dir / timestamp
        
        # Create the directory
        os.makedirs(_CURRENT_RUN_TIMESTAMP_DIR, exist_ok=True)
    
    # Use the existing timestamp directory for this run
    timestamp_dir = _CURRENT_RUN_TIMESTAMP_DIR
    
    # Determine the file prefix based on whether this is a template, error, or formatted prompt
    file_prefix = "template" if is_template else "error" if is_error else "formatted"
    
    # For entity classification prompts, include the column name in the filename
    column_suffix = ""
    if "classify_entities_properties" in template_name and "column_name" in kwargs:
        column_suffix = f"_{kwargs['column_name']}"
    
    # Save the prompt content
    prompt_file = timestamp_dir / f"{template_name.replace('.txt', '')}{column_suffix}_{file_prefix}.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(formatted_prompt)
    
    # Save the kwargs used to format the prompt
    kwargs_file = timestamp_dir / f"{template_name.replace('.txt', '')}{column_suffix}_{file_prefix}_kwargs.json"
    with open(kwargs_file, 'w', encoding='utf-8') as f:
        # Convert any non-serializable objects to strings
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool, list, dict)) and not isinstance(value, type):
                serializable_kwargs[key] = value
            else:
                serializable_kwargs[key] = str(value)
        
        json.dump(serializable_kwargs, f, indent=2)
        
    # Save metadata about the prompt sample if it doesn't exist yet
    metadata_file = timestamp_dir / "metadata.json"
    if not metadata_file.exists():
        metadata = {
            "run_timestamp": timestamp,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "templates": []
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    # Update the metadata file with this template
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Add this template to the list if not already present
        template_info = {
            "template_name": template_name,
            "is_template": is_template,
            "is_error": is_error,
            "file_prefix": file_prefix,
            "node_type": template_name.replace('.txt', ''),
        }
        
        # For entity classification, include the column name in the metadata
        if "classify_entities_properties" in template_name and "column_name" in kwargs:
            template_info["column_name"] = kwargs["column_name"]
        
        # Check if this template is already in the list
        template_exists = False
        for existing in metadata.get("templates", []):
            if (existing.get("template_name") == template_name and 
                existing.get("is_template") == is_template and
                existing.get("is_error") == is_error and
                existing.get("column_name", None) == template_info.get("column_name", None)):
                template_exists = True
                break
        
        if not template_exists:
            if "templates" not in metadata:
                metadata["templates"] = []
            metadata["templates"].append(template_info)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
    except Exception as e:
        # If there's an error updating the metadata, just log it and continue
        print(f"Error updating metadata: {e}")
        pass

def call_llm_with_json_output(prompt: str, state_name: str = None) -> Dict[str, Any]:
    """
    Call the LLM and parse the response as JSON.
    
    Args:
        prompt: The prompt to send to the LLM
        state_name: The name of the state to use the LLM for
        
    Returns:
        The parsed JSON response as a dictionary
    """
    # If no state is specified, use a default state
    if state_name is None:
        state_name = next(iter(LLM_CONFIGS.keys()), None)
    
    # Get the LLM configuration for this state
    state_config = LLM_CONFIGS.get(state_name, {})
    
    # Check if LMStudio is available and should be used
    if LMSTUDIO_AVAILABLE:
        try:
            # Add explicit instructions to return JSON
            json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
            
            # Get LMStudio client
            client = get_lmstudio_client()
            
            # Call the LMStudio API
            logger.debug(f"Calling LMStudio API for state '{state_name}'")
            response = client.completion(json_prompt)
            
            # Extract the text from the response
            response_text = client.extract_completion_text(response)
            
            # Try to parse the response as JSON
            json_data = extract_json_from_llm_response(response_text)
            
            if json_data:
                logger.debug(f"Successfully parsed JSON response from LMStudio for state '{state_name}'")
                return json_data
            else:
                logger.warning(f"Failed to parse JSON response from LMStudio for state '{state_name}'. Response: {response_text}")
                
                # If we couldn't extract JSON, try again with a more explicit prompt
                retry_prompt = f"{prompt}\n\nYou MUST respond with ONLY valid JSON. No other text. No markdown formatting."
                
                # Call the LMStudio API again
                retry_response = client.completion(retry_prompt)
                retry_text = client.extract_completion_text(retry_response)
                
                # Try to parse the retry response
                retry_json = extract_json_from_llm_response(retry_text)
                
                if retry_json:
                    logger.debug(f"Successfully parsed JSON from retry response from LMStudio for state '{state_name}'")
                    return retry_json
                else:
                    logger.error(f"Failed to parse JSON from retry response from LMStudio for state '{state_name}'. Response: {retry_text}")
                    # Return an empty dict as a fallback
                    return {}
                    
        except Exception as e:
            logger.error(f"Error using LMStudio for state '{state_name}': {str(e)}")
            logger.warning("Falling back to default LLM provider")
    
    # If LMStudio is not available or there was an error, use the default implementation
    # Add explicit instructions to return JSON
    json_prompt = f"{prompt}\n\nPlease provide your response in valid JSON format."
    
    # Call the LLM
    response = call_llm_with_state(state_name, json_prompt)
    
    # Try to parse the response as JSON
    try:
        # First try to extract JSON from the response
        json_data = extract_json_from_llm_response(response)
        
        if json_data:
            logger.debug(f"Successfully parsed JSON response for state '{state_name}'")
            return json_data
        else:
            logger.warning(f"Failed to parse JSON response for state '{state_name}'. Response: {response}")
            
            # If we couldn't extract JSON, try again with a more explicit prompt
            retry_prompt = f"{prompt}\n\nYou MUST respond with ONLY valid JSON. No other text. No markdown formatting."
            
            # Call the LLM again
            retry_response = call_llm_with_state(state_name, retry_prompt)
            
            # Try to parse the retry response
            retry_json = extract_json_from_llm_response(retry_response)
            
            if retry_json:
                logger.debug(f"Successfully parsed JSON from retry response for state '{state_name}'")
                return retry_json
            else:
                logger.error(f"Failed to parse JSON from retry response for state '{state_name}'. Response: {retry_response}")
                # Return an empty dict as a fallback
                return {}
    
    except Exception as e:
        logger.error(f"Error parsing JSON response for state '{state_name}': {str(e)}")
        # Return a dictionary with the error and raw response
        return {
            "error": str(e),
            "raw_response": response
        }

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
