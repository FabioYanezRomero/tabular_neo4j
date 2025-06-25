"""
Node for generating semantic embeddings for columns using Ollama embedding models.
This node should be run after columns_contextualization_node and stores embeddings per table/column.

Recommended Ollama embedding models:
- 'nomic-embed-text' (balanced, default)
- 'mxbai-embed-large' (high accuracy)
- 'all-minilm' (high speed, low resource)

This node calls Ollama using the REST API via call_ollama_embed_api utility, which is compatible with containerized environments.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from Tabular_to_Neo4j.app_state import MultiTableGraphState
import os
import requests

# Default model to use; can be made configurable
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")


def call_ollama_embed_api(text: str, model: str = DEFAULT_OLLAMA_MODEL, url: str = OLLAMA_URL) -> List[float]:
    """
    Call the Ollama embedding endpoint via REST API.
    """
    endpoint = f"{url}/api/embeddings"
    payload = {"model": model, "prompt": text}
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception as e:
        raise RuntimeError(f"Ollama embedding API call failed: {e}")


def embed_text(text: str, model: str = DEFAULT_OLLAMA_MODEL) -> List[float]:
    return call_ollama_embed_api(text, model)


def semantic_embedding_node(state: MultiTableGraphState, config: Optional[Dict[str, Any]] = None) -> MultiTableGraphState:
    """
    For each table and column, generate semantic embeddings for contextualized column descriptions.
    Stores results as a matrix per table: state[table]["column_embeddings"] = {
        "columns": [col1, col2, ...],
        "embeddings": np.ndarray (shape: n_columns x embedding_dim)
    }
    """
    model = (config or {}).__getitem__("embedding_model") if (config and "embedding_model" in config) else DEFAULT_OLLAMA_MODEL
    for table_name, table_state in state.items():
        contextualizations = table_state.get("columns_contextualization", [])
        columns = [c["column"] for c in contextualizations]
        texts = [c["contextualization"] for c in contextualizations]
        if not texts:
            continue
        # Parallel embedding
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(lambda txt: embed_text(txt, model), texts))
        # Store as matrix for easy retrieval
        embeddings_matrix = np.vstack(embeddings)
        table_state["column_embeddings"] = {
            "columns": columns,
            "embeddings": embeddings_matrix,
            "model": model
        }
    return state
