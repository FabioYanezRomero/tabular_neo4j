"""
Utility module for saving node outputs to files.
"""

import os
import json
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, MutableMapping

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

class OutputSaver:
    """
    Class for saving node outputs to files in a structured, per-table (or inter_table) directory format.
    Also stores the node order mapping for consistent file prefixing.
    """
    def __init__(self, base_dir: str = "samples"):
        self.base_dir = base_dir
        # Include microseconds to avoid collisions when multiple runs start within the same second
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = os.path.join(self.base_dir, self.timestamp)
        self.previous_state = {}
        self.node_order_map = {}  # node name -> index mapping for current pipeline
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")

    def set_node_order_map(self, node_list):
        """Set the node order mapping for the current pipeline."""
        self.node_order_map = {name: idx + 1 for idx, (name, _) in enumerate(node_list)}
        logger.info(f"Node order mapping set: {self.node_order_map}")

    def get_node_order(self, node_name):
        """Get the node order index for a given node name."""
        return self.node_order_map.get(node_name, 0)

    def save_llm_output_sample(
        self,
        *,
        node_name: str,
        output: dict | list,
        node_order: int = 0,
        table_name: str = None,
        unique_suffix: str = "",
        template_name: str = None,
    ) -> None:
        """
        Save an LLM output sample for a specific call, using a unique suffix and template name.
        The output will be saved in <base_dir>/<timestamp>/<table_name>/llm_outputs/<node_order>_<node_name>_<unique_suffix>.json
        Args:
            node_name: Name of the node/state (e.g., classify_entities_properties)
            output: Output dict (e.g., response from LLM call)
            node_order: Order of the node in the pipeline (for file naming)
            table_name: Table name or "inter_table" for cross-table nodes
            unique_suffix: Suffix for the output file (e.g., column/property/entity pair)
            template_name: Name of the prompt template (optional, for traceability)
        """
        table_dir = self._get_table_dir(table_name or "default")
        llm_outputs_dir = os.path.join(table_dir, "llm_outputs")
        os.makedirs(llm_outputs_dir, exist_ok=True)
        # Compose file name
        # Remove any occurrence of .txt from unique_suffix to avoid .txt.json files
        clean_suffix = unique_suffix.replace('.txt', '') if unique_suffix else unique_suffix
        suffix = f"_{clean_suffix}" if clean_suffix else ""
        template_part = f"_{template_name}" if template_name else ""
        file_name = f"{node_order:02d}_{node_name}{suffix}{template_part}.json"
        # Clean up file name (remove spaces, problematic chars)
        file_name = file_name.replace(" ", "_").replace("/", "-")
        file_name = file_name.replace('.txt', '')
        output_file = os.path.join(llm_outputs_dir, file_name)
        try:
            # Only attempt to repair embedded JSON if output is a dict
            if isinstance(output, dict):
                output = self.clean_embedded_json(output, ["response", "raw_response"])
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(output), f, indent=2)
            logger.info(f"Saved LLM output for node '{node_name}' (order: {node_order}, suffix: '{unique_suffix}') to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save LLM output for node '{node_name}': {str(e)}")

    def _get_table_dir(self, table_name: str) -> str:
        """
        Return the directory path where outputs for *table_name* should be written.
        The layout is now::
            <output_dir>/<dataset>/<table_name>/node_outputs/
            <output_dir>/<dataset>/inter_table/node_outputs/
        A *table_name* is expected to follow the convention  "<dataset>__<table>"
        or "<dataset>__inter_table".  If the separator is not present, we treat the
        entire string as the dataset (legacy behaviour).
        """
        if table_name is None:
            return self.output_dir  # safeguard – should not generally happen

        # Split the synthetic key created in the pipeline
        if "__" in table_name:
            dataset, table = table_name.split("__", 1)
            if table == "inter_table":
                return os.path.join(self.output_dir, dataset, "inter_table")
            return os.path.join(self.output_dir, dataset, table)
        # No separator – legacy single-dataset layout
        return os.path.join(self.output_dir, table_name)

    def save_node_output(self, node_name: str, state: MutableMapping[str, Any], node_order: int = 0, table_name: str = None) -> None:
        """
            Save the output of a node to a file, organized by table (or inter_table).
        Args:
            node_name: Name of the node
            state: Current state of the graph
            node_order: Order of the node in the pipeline (for file naming)
            table_name: Table name or "inter_table" for cross-table nodes
        """
        table_dir = self._get_table_dir(table_name or "default")
        node_outputs_dir = os.path.join(table_dir, "node_outputs")
        os.makedirs(node_outputs_dir, exist_ok=True)
        # Track previous state per table/inter_table
        # PATCH: Always write the full state for each node, not just the diff
        serializable_state = self._make_serializable(state)
        output_file = os.path.join(node_outputs_dir, f"{node_order:02d}_{node_name}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            logger.debug(f"[PATCHED] Saved full state from node '{node_name}' (order: {node_order}) to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output of node '{node_name}': {str(e)}")
        # Update previous state for this table/inter_table
        self.previous_state[table_name] = state.copy()

    def _extract_new_info(self, current_state: MutableMapping[str, Any], prev_state: MutableMapping[str, Any] | None = None) -> Dict[str, Any]:
        """
        Extract only the new or changed information from the current state, using table-specific previous state.
        """
        new_info = {}
        if prev_state is None:
            return current_state
        for key, value in current_state.items():
            if key not in prev_state:
                new_info[key] = value
            elif self._is_different(value, prev_state[key]):
                new_info[key] = value
        return new_info
    
    def _is_different(self, value1: Any, value2: Any) -> bool:
        import numpy as np
        import pandas as pd
        logger.debug(f"Comparing {type(value1)} and {type(value2)}")
        
        # Handle pandas DataFrames
        if isinstance(value1, pd.DataFrame) and isinstance(value2, pd.DataFrame):
            try:
                result = not value1.equals(value2)
                logger.debug(f"DataFrame comparison result: {result}")
                return result
            except Exception as e:
                logger.warning(f"DataFrame comparison failed: {e}. Treating as different.")
                return True
        
        # Handle pandas Series
        if isinstance(value1, pd.Series) and isinstance(value2, pd.Series):
            try:
                result = not value1.equals(value2)
                logger.debug(f"Series comparison result: {result}")
                return result
            except Exception as e:
                logger.warning(f"Series comparison failed: {e}. Treating as different.")
                return True
        
        # Handle numpy arrays
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            result = not np.array_equal(value1, value2)
            logger.debug(f"NumPy array comparison result: {result}")
            return result
        
        # Fallback for all other types
        try:
            result = value1 != value2
            logger.debug(f"Comparison result for {type(value1)}: {result}")
            if not isinstance(result, bool):
                logger.warning(f"Comparison returned non-bool value {result} for {type(value1)}. Treating as different.")
                return True
            return result
        except Exception as e:
            logger.warning(f"Comparison failed for {type(value1)}: {e}. Treating as different.")
            return True
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to a dictionary representation
            return {
                "type": "DataFrame",
                "columns": obj.columns.tolist(),
                "data": obj.values.tolist(),
                "index": obj.index.tolist()
            }
        elif isinstance(obj, pd.Series):
            return {
                "type": "Series",
                "data": obj.tolist(),
                "index": obj.index.tolist()
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to string representation
            try:
                return str(obj)
            except:
                return f"<Unserializable object of type {type(obj).__name__}>"

    @staticmethod
    def clean_embedded_json(json_obj, key_path):
        """
        Cleans and parses an embedded JSON string at the specified key path.
        If the value at the path is a string containing malformed JSON (e.g., missing closing brackets),
        attempts to repair and parse it, replacing the value with the parsed object if successful.
        """
        import json
        import logging
        # Traverse to the target field
        target = json_obj
        for key in key_path[:-1]:
            if not isinstance(target, dict):
                return json_obj
            target = target.get(key, {})
        last_key = key_path[-1]

        if not isinstance(target, dict):
            return json_obj  # Can't get the last key if not a dict
        raw_json_str = target.get(last_key)
        if not isinstance(raw_json_str, str):
            return json_obj  # Nothing to do

        # Try to parse as is
        try:
            parsed = json.loads(raw_json_str)
            target[last_key] = parsed
            return json_obj
        except json.JSONDecodeError:
            # Try to fix common issues (e.g., missing closing brackets)
            fixed_str = raw_json_str
            # Count brackets to guess what's missing
            open_curly = fixed_str.count('{')
            close_curly = fixed_str.count('}')
            open_square = fixed_str.count('[')
            close_square = fixed_str.count(']')
            # Add missing closing brackets if needed
            if open_square > close_square:
                fixed_str += ']' * (open_square - close_square)
            if open_curly > close_curly:
                fixed_str += '}' * (open_curly - close_curly)
            try:
                parsed = json.loads(fixed_str)
                target[last_key] = parsed
            except json.JSONDecodeError as e:
                logging.warning(f"Could not fix embedded JSON: {e}")
            return json_obj

# Global instance for use across the application
output_saver = None

def initialize_output_saver(base_dir: str = "samples") -> OutputSaver:
    """
    Initialize the global output saver instance.
    
    Args:
        base_dir: Base directory for saving outputs
        
    Returns:
        OutputSaver instance
    """
    global output_saver
    output_saver = OutputSaver(base_dir)
    return output_saver

