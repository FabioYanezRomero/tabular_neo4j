import json
import logging
from pathlib import Path
from Tabular_to_Neo4j.app_state import GraphState

def load_column_analytics_node(state: GraphState, node_order: int, use_analytics: bool = False) -> GraphState:
    """
    Load precomputed column analytics from JSON files into state['column_analytics'].
    """
    logger = logging.getLogger(__name__)
    if not use_analytics:
        logger.debug("use_analytics is False; skipping loading column analytics")
        return state

    csv_path = state.get("csv_file_path")
    if not csv_path:
        logger.warning("No 'csv_file_path' in state; cannot load column analytics")
        return state

    path = Path(csv_path)
    # Infer dataset name assuming path contains 'csvs'
    try:
        parts = path.parts
        idx = parts.index("csvs")
        dataset_name = parts[idx + 1]
    except ValueError:
        logger.warning(
            "Cannot infer dataset name from path %s; expected '/.../csvs/<dataset>/<table>.csv'", csv_path
        )
        return state

    table_name = path.stem
    analytics_dir = Path("/app/analytics/csvs") / dataset_name / table_name
    analytics = {}
    if analytics_dir.exists() and analytics_dir.is_dir():
        for f in analytics_dir.glob("*.json"):
            try:
                with f.open("r", encoding="utf-8") as jf:
                    analytics[f.stem] = json.load(jf)
            except Exception as e:
                logger.error("Failed to load analytics file %s: %s", f, e)
    else:
        logger.warning(
            "Analytics directory %s not found; skipping analytics load", analytics_dir
        )

    state["column_analytics"] = analytics
    logger.info("Loaded analytics for %d columns from %s", len(analytics), analytics_dir)
    return state
