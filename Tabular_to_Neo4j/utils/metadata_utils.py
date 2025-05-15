"""
Utility functions for handling metadata files.
These utilities help load and format metadata for use in LLM prompts.
"""

import os
import json
from typing import Dict, Any, Optional, List
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def get_metadata_path_for_csv(csv_file_path: str) -> str:
    """
    Get the path to the metadata file for a given CSV file path.
    The metadata file should be in the /app/Tabular_to_Neo4j/sample_data/metadata directory
    with the same name as the CSV file but with a .json extension.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Path to the corresponding metadata file
    """
    # Get the file name without extension
    file_name = os.path.basename(csv_file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    # Construct the metadata path using the specified directory structure
    metadata_path = os.path.join('/app/Tabular_to_Neo4j/sample_data/metadata', f"{file_name_without_ext}.json")
    
    logger.debug(f"Metadata path for {csv_file_path}: {metadata_path}")
    return metadata_path

def load_metadata_for_csv(csv_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata for a given CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary containing metadata or None if metadata file doesn't exist
    """
    metadata_path = get_metadata_path_for_csv(csv_file_path)
    
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
                return metadata
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading metadata file: {str(e)}")
        return None

def format_metadata_for_prompt(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into a string suitable for inclusion in LLM prompts.
    Simply converts the raw metadata to a JSON string.
    
    Args:
        metadata: Dictionary containing metadata
        
    Returns:
        Formatted metadata string for LLM prompts
    """
    if not metadata:
        return "No metadata available."
    
    # Convert metadata to a formatted JSON string
    import json
    metadata_str = json.dumps(metadata, indent=2)
    
    # Return just the JSON string without any prefix
    return metadata_str

def get_metadata_for_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Helper function to get metadata from a state dictionary.
    
    Args:
        state: State dictionary that may contain a CSV file path
        
    Returns:
        Metadata dictionary or None if not available
    """
    csv_file_path = state.get('csv_file_path', '')
    if not csv_file_path:
        logger.warning("No CSV file path in state, cannot load metadata")
        return None
    
    return load_metadata_for_csv(csv_file_path)
