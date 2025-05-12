"""
Utility functions for LLM interactions.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Union
import requests
from pathlib import Path
from Tabular_to_Neo4j.utils.logging_config import get_logger
import re

# Import configuration
from Tabular_to_Neo4j.config import (

# Configure logging
logger = get_logger(__name__)

    LLM_API_KEY, 
    LLM_PROVIDER, 
    LLM_MODEL_NAME_GENERAL,
    LLM_MODEL_NAME_TRANSLATE,
    LMSTUDIO_BASE_URL,
    HUGGINGFACE_MODEL_ID
)

# Set up logging
logging.basicConfig(level=logging.INFO)


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

def call_openai_api(prompt: str, model: str = LLM_MODEL_NAME_GENERAL, max_retries: int = 3) -> str:
    """
    Call the OpenAI API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
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
                temperature=0.1,  # Low temperature for more deterministic responses
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

def call_lmstudio_api(prompt: str, model: str = None, max_retries: int = 3) -> str:
    """
    Call the LMStudio API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
        model: Not used for LMStudio, but kept for API consistency
        max_retries: Maximum number of retries on failure
        
    Returns:
        The LLM response as a string
    """
    base_url = LMSTUDIO_BASE_URL
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"LMStudio API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LMStudio API call failed after {max_retries} attempts: {e}")
                raise

def call_huggingface_api(prompt: str, model: str = HUGGINGFACE_MODEL_ID, max_retries: int = 3) -> str:
    """
    Call the Hugging Face API with retry logic.
    
    Args:
        prompt: The prompt to send to the API
        model: The model ID to use
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
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json={"inputs": prompt, "parameters": {"temperature": 0.1}}
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

def call_llm(prompt: str, model: str = None, is_translation: bool = False) -> str:
    """
    Call the configured LLM provider.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Override the default model if specified
        is_translation: Whether this is a translation task (uses translation model)
        
    Returns:
        The LLM response as a string
    """
    if not model:
        model = LLM_MODEL_NAME_TRANSLATE if is_translation else LLM_MODEL_NAME_GENERAL
    
    if LLM_PROVIDER.lower() == "openai":
        return call_openai_api(prompt, model)
    elif LLM_PROVIDER.lower() == "lmstudio":
        return call_lmstudio_api(prompt, model)
    elif LLM_PROVIDER.lower() == "huggingface":
        return call_huggingface_api(prompt, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

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

def call_llm_with_json_output(prompt: str, model: str = None, is_translation: bool = False) -> Dict[str, Any]:
    """
    Call the LLM and parse the response as JSON.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Override the default model if specified
        is_translation: Whether this is a translation task
        
    Returns:
        The parsed JSON response as a dictionary
    """
    response = call_llm(prompt, model, is_translation)
    return extract_json_from_llm_response(response)
