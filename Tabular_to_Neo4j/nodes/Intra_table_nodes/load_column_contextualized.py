import logging
from pathlib import Path
from Tabular_to_Neo4j.app_state import GraphState

def load_column_contextualized_node(state: GraphState, node_order: int, **_) -> GraphState:  # noqa: D401
    """Load contextualized analytics plain-text descriptions into state["column_analytics"].

    The descriptions are produced by *precompute_column_analytics_dask.py* and stored under
    ``/app/contextualized_analytics/<dataset>/<table>/<column>.txt``.
    Each file contains a natural-language summary for a column.  We load the raw text so that
    downstream LLM prompts can directly condition on it.
    """

    logger = logging.getLogger(__name__)

    csv_path = state.get("csv_file_path")
    if not csv_path:
        logger.warning("No 'csv_file_path' in state; cannot load contextualized analytics")
        return state

    path = Path(csv_path)
    try:
        parts = path.parts
        idx = parts.index("csvs")
        dataset_name = parts[idx + 1]
    except ValueError:
        logger.warning("Cannot infer dataset name from %s; skipping contextualized analytics load", csv_path)
        return state

    table_name = path.stem
    ctx_dir = Path("/app/contextualized_analytics") / dataset_name / table_name
    analytics: dict[str, str] = {}
    if ctx_dir.exists():
        for txt in ctx_dir.glob("*.txt"):
            try:
                analytics[txt.stem] = txt.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.error("Failed reading %s: %s", txt, exc)
    else:
        logger.info("Contextualized analytics directory %s not found", ctx_dir)

    state["column_analytics"] = analytics
    logger.info("Loaded contextualized analytics for %d columns from %s", len(analytics), ctx_dir)
    return state
