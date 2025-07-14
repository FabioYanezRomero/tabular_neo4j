"""Inter-table node that computes semantic similarity between columns across tables.

Steps:
1. For every column in every table, build a contextualized text description
   (using generate_text_sequence from columns_contextualization).
2. Generate embeddings with Ollama (via call_ollama_embed_api).
3. Compute cosine similarity for every cross-table pair.
4. Store pairs whose similarity >= SIM_THRESHOLD in global state key
   `cross_table_column_similarity` as a list of dicts.

Assumptions:
- `state` is MultiTableGraphState; each table GraphState has `column_analytics`.
- Analytics may have been precomputed and loaded; if missing, node skips table.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.nodes.cross_table_analysis.columns_contextualization import generate_text_sequence
from Tabular_to_Neo4j.nodes.cross_table_analysis.semantic_embedding_node import call_ollama_embed_api, cosine_similarity

logger = logging.getLogger(__name__)

SIM_THRESHOLD = 0.85  # configurable
EMBED_MODEL = "nomic-embed-text"


def _contextualize_columns(state: MultiTableGraphState) -> List[Tuple[str, str, str]]:
    """Return list of (table, column, contextualized_text)."""
    out: List[Tuple[str, str, str]] = []
    for table_name, tbl_state in state.items():
        if not isinstance(tbl_state, GraphState):
            tbl_state = GraphState.from_dict(dict(tbl_state))
            state[table_name] = tbl_state  # replace with proper object
        analytics = tbl_state.get("column_analytics") or {}
        for col_name, col_meta in analytics.items():
            text = generate_text_sequence(col_meta | {"column_name": col_name})
            out.append((table_name, col_name, text))
    return out


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Parallel embedding via Ollama."""
    embeddings: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(call_ollama_embed_api, txt, EMBED_MODEL): idx for idx, txt in enumerate(texts)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                embeddings[idx] = fut.result()
            except Exception as e:
                logger.error("Embedding failed for index %d: %s", idx, e)
                embeddings[idx] = []
    return embeddings


def column_semantic_similarity_node(state: MultiTableGraphState, node_order: int):  # type: ignore[type-arg]
    logger.info("[column_semantic_similarity_node] Starting computation")

    col_texts = _contextualize_columns(state)
    if not col_texts:
        logger.warning("No columns found for similarity computation")
        state["cross_table_column_similarity"] = []
        return state

    texts = [t[2] for t in col_texts]
    embeddings = _embed_texts(texts)

    # Build list of pairs above threshold
    similar_pairs: List[Dict[str, Any]] = []
    n = len(col_texts)
    for i in range(n):
        table_i, col_i, _ = col_texts[i]
        emb_i = embeddings[i]
        if not emb_i:
            continue
        for j in range(i + 1, n):
            table_j, col_j, _ = col_texts[j]
            if table_i == table_j:
                continue  # only cross-table
            emb_j = embeddings[j]
            if not emb_j:
                continue
            sim = cosine_similarity(emb_i, emb_j)
            if sim >= SIM_THRESHOLD:
                similar_pairs.append({
                    "table_a": table_i,
                    "column_a": col_i,
                    "table_b": table_j,
                    "column_b": col_j,
                    "similarity": sim,
                })
    # store
    state["cross_table_column_similarity"] = similar_pairs
    logger.info("[column_semantic_similarity_node] Stored %d similar column pairs", len(similar_pairs))
    return state
