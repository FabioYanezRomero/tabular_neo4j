import os
import json
from pathlib import Path

def save_state_snapshot(graph_state, timestamp=None, base_dir=None):
    """
    Save each key of the final GraphState as a separate JSON file inside samples/<timestamp>/state/
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

    for key, value in graph_state.items():
        # Try to serialize, fallback to str
        try:
            with open(state_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, default=str)
        except Exception as e:
            with open(state_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump({"error": f"Could not serialize: {e}", "value": str(value)}, f, indent=2)

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
