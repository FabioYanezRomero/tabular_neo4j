import logging
from pathlib import Path
from Tabular_to_Neo4j.app_state import GraphState

def load_column_samples_node(state: GraphState, node_order: int, **_) -> GraphState:  # noqa: D401
    """Load per-column sample strings from /app/no_analytics.

    Files are located at /app/no_analytics/<dataset>/<table>/<column>.txt and contain
    comma-separated sample values. The samples are stored in ``state['column_samples']``
    as a dict {column: [values...]}. This allows downstream prompts to reference raw
    values when full analytics are unavailable.
    """
    logger = logging.getLogger(__name__)

    csv_path = state.get("csv_file_path")
    if not csv_path:
        logger.warning("No 'csv_file_path' in state; cannot load column samples")
        return state

    path = Path(csv_path)
    try:
        parts = path.parts
        idx = parts.index("csvs")
        dataset_name = parts[idx + 1]
    except ValueError:
        logger.warning("Cannot infer dataset name from %s; skipping samples load", csv_path)
        return state

    table_name = path.stem
    samples_dir = Path("/app/no_analytics") / dataset_name / table_name
    samples: dict[str, list[str]] = {}
    if samples_dir.exists():
        for txt in samples_dir.glob("*.txt"):
            try:
                raw = txt.read_text(encoding="utf-8").strip()
                samples[txt.stem] = [s.strip() for s in raw.split(",") if s.strip()]
            except Exception as exc:
                logger.error("Failed reading %s: %s", txt, exc)
    else:
        logger.info("Samples directory %s not found", samples_dir)

    state["column_samples"] = samples
    logger.info("Loaded samples for %d columns from %s", len(samples), samples_dir)
    return state
