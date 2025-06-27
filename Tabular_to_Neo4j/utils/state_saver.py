import os
import json
from pathlib import Path

def save_state_snapshot(graph_state, timestamp=None, base_dir=None):
    """
    Save each key of the final GraphState as a separate JSON file inside samples/<timestamp>/<table_name>/state/ for multi-table, or /inter_table/state/ for inter-table state.
    """
    from Tabular_to_Neo4j.utils.output_saver import get_output_saver
    output_saver = get_output_saver()
    if base_dir is None:
        base_dir = output_saver.base_dir if output_saver else "samples"
    if timestamp is None:
        timestamp = output_saver.timestamp if output_saver else None
    if not timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    from collections.abc import Mapping
    # If this is a MultiTableGraphState (dict-like), save each table's state separately
    if isinstance(graph_state, dict) and all(isinstance(v, Mapping) for v in graph_state.values()):
        for table_name, table_state in graph_state.items():
            state_dir = Path(base_dir) / timestamp / table_name / "state"
            os.makedirs(state_dir, exist_ok=True)
            try:
                with open(state_dir / f"state.json", "w", encoding="utf-8") as f:
                    json.dump(table_state, f, indent=2, default=str)
            except Exception as e:
                with open(state_dir / f"state.json", "w", encoding="utf-8") as f:
                    json.dump({"error": f"Could not serialize: {e}", "value": str(table_state)}, f, indent=2)
    elif isinstance(graph_state, dict) and ("inter_table" in str(graph_state) or "cross_table" in str(graph_state)):
        # Save inter_table state
        state_dir = Path(base_dir) / timestamp / "inter_table" / "state"
        os.makedirs(state_dir, exist_ok=True)
        try:
            with open(state_dir / "state.json", "w", encoding="utf-8") as f:
                json.dump(graph_state, f, indent=2, default=str)
        except Exception as e:
            with open(state_dir / "state.json", "w", encoding="utf-8") as f:
                json.dump({"error": f"Could not serialize: {e}", "value": str(graph_state)}, f, indent=2)
    else:
        # Single-table or fallback (require table_name argument in future)
        raise ValueError("save_state_snapshot requires a table_name or MultiTableGraphState with table names as keys.")


def save_dynamic_config(state_name, config, timestamp=None, base_dir=None):
    """
    Save the resolved LLM config for a dynamic state as a JSON file in samples/<timestamp>/state/
    """
    from Tabular_to_Neo4j.utils.output_saver import get_output_saver
    output_saver = get_output_saver()
    if base_dir is None:
        base_dir = output_saver.base_dir if output_saver else "samples"
    if timestamp is None:
        timestamp = output_saver.timestamp if output_saver else None
    if not timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_dir = Path(base_dir) / timestamp / "state"
    os.makedirs(state_dir, exist_ok=True)
    with open(state_dir / f"{state_name}.config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
