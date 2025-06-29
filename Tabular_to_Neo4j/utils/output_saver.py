"""
Utility module for saving node outputs to files.
"""

import os
import json
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

class OutputSaver:
    """
    Class for saving node outputs to files in a structured, per-table (or inter_table) directory format.
    Also stores the node order mapping for consistent file prefixing.
    """
    def __init__(self, base_dir: str = "samples"):
        self.base_dir = base_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    """
    Class for saving node outputs to files in a structured, per-table (or inter_table) directory format.
    """
    
    # ...existing methods...

    def save_llm_output_sample(
        self,
        *,
        node_name: str,
        output: dict,
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
        suffix = f"_{unique_suffix}" if unique_suffix else ""
        template_part = f"_{template_name}" if template_name else ""
        file_name = f"{node_order:02d}_{node_name}{suffix}{template_part}.json"
        # Clean up file name (remove spaces, problematic chars)
        file_name = file_name.replace(" ", "_").replace("/", "-")
        output_file = os.path.join(llm_outputs_dir, file_name)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(output), f, indent=2)
            logger.info(f"Saved LLM output for node '{node_name}' (order: {node_order}, suffix: '{unique_suffix}') to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save LLM output for node '{node_name}': {str(e)}")

    """
    Class for saving node outputs to files in a structured, per-table (or inter-table) directory format.
    """
    
    def __init__(self, base_dir: str = "samples"):
        """
        Initialize the OutputSaver.
        Args:
            base_dir: Base directory for saving outputs
        """
        self.base_dir = base_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.base_dir, self.timestamp)
        # Track previous_state per table/inter_table
        self.previous_state = {}
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")

    def _get_table_dir(self, table_name: str = None):
        """Return the directory for a given table or inter_table."""
        if table_name is None:
            return self.output_dir  # fallback for legacy/single-table
        return os.path.join(self.output_dir, table_name)

    def save_node_output(self, node_name: str, state: Dict[str, Any], node_order: int = 0, table_name: str = None) -> None:
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
        prev_state = self.previous_state.get(table_name, None)
        new_info = self._extract_new_info(state, prev_state)
        serializable_new_info = self._make_serializable(new_info)
        output_file = os.path.join(node_outputs_dir, f"{node_order:02d}_{node_name}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_new_info, f, indent=2)
            logger.debug(f"Saved new output from node '{node_name}' (order: {node_order}) to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output of node '{node_name}': {str(e)}")
        # Update previous state for this table/inter_table
        self.previous_state[table_name] = state.copy()
    
    def _extract_new_info(self, current_state: Dict[str, Any], prev_state: Dict[str, Any] = None) -> Dict[str, Any]:
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
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to string representation
            try:
                return str(obj)
            except:
                return f"<Unserializable object of type {type(obj).__name__}>"

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

