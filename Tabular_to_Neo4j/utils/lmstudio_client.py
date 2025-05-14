"""
LMStudio client module for interacting with the LMStudio API.
"""

import json
import requests
from typing import Dict, Any, List, Optional, Union
import logging
from Tabular_to_Neo4j.config.lmstudio_config import (
    LMSTUDIO_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P
)

# Configure logging
logger = logging.getLogger(__name__)

class LMStudioClient:
    """Client for interacting with LMStudio API."""
    
    def __init__(
        self,
        api_base: str = LMSTUDIO_ENDPOINT,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P
    ):
        """
        Initialize the LMStudio client.
        
        Args:
            api_base: Base URL for the LMStudio API
            model: Model to use for completions
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
        """
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Check if LMStudio is available and get available models
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Check if LMStudio is available and get available models."""
        try:
            response = requests.get(f"{self.api_base}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Connected to LMStudio. Available models: {models}")
                
                # Use the first available model if the default model is not available
                if models and isinstance(models, list) and models:
                    self.model = models[0].get('id', DEFAULT_MODEL)
                    logger.info(f"Using model: {self.model}")
            else:
                logger.warning(f"Failed to connect to LMStudio. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to LMStudio: {e}")
            raise ConnectionError(f"Failed to connect to LMStudio at {self.api_base}: {e}")
    
    def completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt to generate completions for
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            The completion response
        """
        url = f"{self.api_base}/completions"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "top_p": top_p if top_p is not None else self.top_p,
        }
        
        if stop:
            payload["stop"] = stop
        
        logger.debug(f"Sending completion request to LMStudio: {payload}")
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in LMStudio completion request: {e}")
            raise
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion for the given messages.
        
        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            The chat completion response
        """
        url = f"{self.api_base}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "top_p": top_p if top_p is not None else self.top_p,
        }
        
        if stop:
            payload["stop"] = stop
        
        logger.debug(f"Sending chat completion request to LMStudio: {payload}")
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in LMStudio chat completion request: {e}")
            raise
    
    def extract_completion_text(self, response: Dict[str, Any]) -> str:
        """
        Extract the completion text from a completion response.
        
        Args:
            response: The completion response
            
        Returns:
            The completion text
        """
        if "choices" in response and response["choices"]:
            return response["choices"][0].get("text", "")
        return ""
    
    def extract_chat_completion_text(self, response: Dict[str, Any]) -> str:
        """
        Extract the completion text from a chat completion response.
        
        Args:
            response: The chat completion response
            
        Returns:
            The completion text
        """
        if "choices" in response and response["choices"]:
            message = response["choices"][0].get("message", {})
            return message.get("content", "")
        return ""


def get_lmstudio_client() -> LMStudioClient:
    """
    Get an instance of the LMStudio client.
    
    Returns:
        LMStudioClient instance
    """
    return LMStudioClient()
