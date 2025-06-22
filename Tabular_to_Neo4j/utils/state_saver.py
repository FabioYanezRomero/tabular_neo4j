import os
import json
from pathlib import Path

def save_state_snapshot(graph_state, timestamp):
    """
    Save each key of the final GraphState as a separate JSON file inside state/<timestamp>/
    """
    base_dir = Path(__file__).parent.parent.parent
    state_dir = base_dir / "state" / timestamp
    os.makedirs(state_dir, exist_ok=True)

    for key, value in graph_state.items():
        # Try to serialize, fallback to str
        try:
            with open(state_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, default=str)
        except Exception as e:
            with open(state_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump({"error": f"Could not serialize: {e}", "value": str(value)}, f, indent=2)

def save_dynamic_config(state_name, config, timestamp):
    """
    Save the resolved LLM config for a dynamic state as a JSON file in state/<timestamp>/
    """
    base_dir = Path(__file__).parent.parent.parent
    state_dir = base_dir / "state" / timestamp
    os.makedirs(state_dir, exist_ok=True)
    with open(state_dir / f"{state_name}.config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
