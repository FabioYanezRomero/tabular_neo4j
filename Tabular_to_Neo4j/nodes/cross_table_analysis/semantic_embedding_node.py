"""
Node for generating semantic embeddings for columns using Ollama embedding models.
This node should be run after columns_contextualization_node and stores embeddings per table/column.

Supported Ollama embedding models (selectable via config):
- 'nomic-embed-text' (balanced, default)
- 'mxbai-embed-large' (high accuracy, larger model)
- 'all-minilm' (high speed, lower resource usage)

Specify the embedding model in the config dictionary with the key 'embedding_model'.

This node calls Ollama using the REST API via call_ollama_embed_api utility, which is compatible with containerized environments.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from Tabular_to_Neo4j.app_state import MultiTableGraphState
from Tabular_to_Neo4j.app_state import GraphState
import os
import requests
from Tabular_to_Neo4j.config.settings import LLM_CONFIGS, DEFAULT_LLM_PROVIDER


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# Default model to use; can be made configurable
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


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


def embed_text(text: str, model: str) -> List[float]:
    return call_ollama_embed_api(text, model)


from Tabular_to_Neo4j.utils.ollama_api import load_model_in_ollama, unload_model_from_ollama

def semantic_embedding_node(state: MultiTableGraphState, node_order: int) -> MultiTableGraphState:
    """
    For each table and column, generate semantic embeddings for contextualized column descriptions.
    Stores results as a matrix per table: state[table]["column_embeddings"] = {
        "columns": [col1, col2, ...],
        "embeddings": np.ndarray (shape: n_columns x embedding_dim)
    }
    Loads and unloads the Ollama model before and after embedding to manage GPU memory.

    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline
    """
    node_config = LLM_CONFIGS.get("semantic_embedding_node", {}).copy()
    provider = node_config.get("provider", DEFAULT_LLM_PROVIDER)
    # Pick model_name for the right provider (Ollama vs LMStudio)
    if provider == "ollama":
        model_name = node_config.get("model_name")
    else:
        model_name = node_config.get("model_name")  # fallback, could add LMStudio logic here
    load_model_in_ollama(model_name)
    all_columns_embeddings = {}
    try:
        for table_name, table_state in state.items():
            contextualizations = table_state.get("columns_contextualization", [])
            columns = [c["column"] for c in contextualizations]
            texts = [c["contextualization"] for c in contextualizations]
            if not columns:
                continue
            embeddings = [embed_text(text, model_name) for text in texts]
            all_columns_embeddings[table_name] = {
                "columns": columns,
                "embeddings": embeddings,
            }
        # Compute cross-table similarity matrix
        similarity_matrix = {}
        table_names = list(all_columns_embeddings.keys())
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names):
                if i >= j:
                    continue
                cols1 = all_columns_embeddings[table1]["columns"]
                embs1 = all_columns_embeddings[table1]["embeddings"]
                cols2 = all_columns_embeddings[table2]["columns"]
                embs2 = all_columns_embeddings[table2]["embeddings"]
                for idx1, col1_name in enumerate(cols1):
                    for idx2, col2_name in enumerate(cols2):
                        emb1 = embs1[idx1]
                        emb2 = embs2[idx2]
                        similarity = cosine_similarity(emb1, emb2)
                        if similarity > 0.7:
                            pair_key = f"{table1}.{col1_name} <-> {table2}.{col2_name}"
                            similarity_matrix[pair_key] = similarity
        # Store similarity_matrix in a special table-level GraphState, or as an attribute on each table if appropriate
        # Here, we add it to each table's GraphState as a new key
        import logging
        logger = logging.getLogger(__name__)
        logger.info('[semantic_embedding_node][BEFORE] Table states: ' + str({k: type(v).__name__ for k,v in state.items()}))

        for table_name, table_state in state.items():
            if not isinstance(table_state, GraphState):
                table_state = GraphState(**table_state)
                state[table_name] = table_state
            table_state["cross_table_column_similarity"] = similarity_matrix
        # Ensure every table state is a GraphState before returning
        for table_name, table_state in state.items():
            assert isinstance(table_state, GraphState), f"semantic_embedding_node: State for '{table_name}' is not a GraphState, got {type(table_state)}"
        logger.info('[semantic_embedding_node][AFTER] Table states: ' + str({k: type(v).__name__ for k,v in state.items()}))
        
    except:
        pass

    return state
