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
    Class for saving node outputs to files in a structured directory format.
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
        self.previous_state = {}
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
    
    def save_node_output(self, node_name: str, state: Dict[str, Any], node_order: int = 0) -> None:
        """
        Save the output of a node to a file.
        
        Args:
            node_name: Name of the node
            state: Current state of the graph
            node_order: Order of the node in the pipeline (for file naming)
        """
        # Extract only the new or changed information from this node
        new_info = self._extract_new_info(state)
        
        # Create a serializable copy of the new information
        serializable_new_info = self._make_serializable(new_info)
        
        # Save the new information to a file with order prefix
        output_file = os.path.join(self.output_dir, f"{node_order:02d}_{node_name}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_new_info, f, indent=2)
            logger.debug(f"Saved new output from node '{node_name}' (order: {node_order}) to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output of node '{node_name}': {str(e)}")
        
        # Update the previous state for the next node
        self.previous_state = state.copy()
    
    def _extract_new_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the new or changed information from the current state.
        
        Args:
            current_state: Current state of the graph
            
        Returns:
            Dictionary containing only the new or changed information
        """
        new_info = {}
        
        # If this is the first node, return the entire state
        if not self.previous_state:
            return current_state
        
        # Compare each key in the current state with the previous state
        for key, value in current_state.items():
            # If the key is new or the value has changed, include it in the new info
            if key not in self.previous_state or self._is_different(value, self.previous_state[key]):
                new_info[key] = value
        
        return new_info
    
    def _is_different(self, value1: Any, value2: Any) -> bool:
        """
        Check if two values are different.
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            True if the values are different, False otherwise
        """
        # Handle pandas DataFrames separately
        if isinstance(value1, pd.DataFrame) and isinstance(value2, pd.DataFrame):
            try:
                return not value1.equals(value2)
            except:
                return True
        
        # For other types, use direct comparison
        return value1 != value2
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
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

def get_output_saver() -> Optional[OutputSaver]:
    """
    Get the global output saver instance.
    
    Returns:
        OutputSaver instance or None if not initialized
    """
    return output_saver
