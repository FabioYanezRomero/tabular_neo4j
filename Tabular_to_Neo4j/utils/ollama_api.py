"""
Helpers for loading and unloading models in Ollama via its REST API.
"""
import os
import requests
from Tabular_to_Neo4j.utils.logging_config import get_logger

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
logger = get_logger(__name__)

def load_model_in_ollama(model_name: str, max_retries: int = 3) -> bool:
    """
    Load a model in Ollama (pulls if not present).
    """
    url = f"{OLLAMA_URL}/api/pull"
    payload = {"name": model_name}
    for attempt in range(max_retries):
        try:
            logger.info(f"Pulling model '{model_name}' in Ollama (attempt {attempt+1})...")
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            logger.info(f"Model '{model_name}' loaded in Ollama.")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to load model '{model_name}' in Ollama: {e}. Retrying...")
            else:
                logger.error(f"Failed to load model '{model_name}' in Ollama after {max_retries} attempts: {e}")
    return False

def unload_model_from_ollama(model_name: str, max_retries: int = 3) -> bool:
    """
    Unload (delete) a model from Ollama.
    """
    url = f"{OLLAMA_URL}/api/delete"
    payload = {"name": model_name}
    for attempt in range(max_retries):
        try:
            logger.info(f"Deleting model '{model_name}' from Ollama (attempt {attempt+1})...")
            response = requests.delete(url, json=payload, timeout=30)
            if response.status_code == 200:
                logger.info(f"Model '{model_name}' deleted from Ollama.")
                return True
            else:
                logger.warning(f"Failed to delete model '{model_name}' from Ollama: {response.text}")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to delete model '{model_name}' from Ollama: {e}. Retrying...")
            else:
                logger.error(f"Failed to delete model '{model_name}' from Ollama after {max_retries} attempts: {e}")
    return False
