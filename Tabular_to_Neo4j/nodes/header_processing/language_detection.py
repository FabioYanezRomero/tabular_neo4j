"""
Language detection module for the Tabular to Neo4j converter.
This module handles detecting the language of headers and metadata.
"""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.language_utils import verify_header_language, detect_language, normalize_language_name, are_languages_matching
from Tabular_to_Neo4j.utils.metadata_utils import load_metadata_for_csv
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def get_metadata_language(state: GraphState) -> Tuple[str, float]:
    """
    Get the language from the metadata file associated with the CSV file.
    If metadata file doesn't exist or doesn't specify language, fallback to detection.
    
    Args:
        state: The current graph state
        
    Returns:
        Tuple of (metadata_language, confidence)
    """
    # Get the CSV file path from the state
    csv_file_path = state.get('csv_file_path', '')
    if not csv_file_path:
        logger.warning("No CSV file path in state, cannot load metadata")
        return "en", 0.5  # Default to English with low confidence
    
    # Load metadata for the CSV file
    metadata = load_metadata_for_csv(csv_file_path)
    
    # If metadata exists and contains language information, use it
    if metadata and 'language' in metadata:
        language = metadata['language']
        logger.info(f"Found language in metadata file: {language}")
        return language, 1.0  # High confidence since it's explicitly specified
    
    # If no metadata or no language in metadata, try to detect from file name
    file_name = os.path.basename(csv_file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    file_name_text = file_name_without_ext.replace('_', ' ').replace('-', ' ')
    
    try:
        language, confidence = detect_language(file_name_text)
        logger.debug(f"Detected metadata language from file name: {language} (confidence: {confidence:.2f})")
        return language, confidence
    except Exception as e:
        logger.warning(f"Could not detect language from file name: {e}")
        # Default to English if detection fails
        return "en", 0.5

def detect_header_language_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Detect the language of headers and determine if translation is needed.
    This node checks if the header language matches the metadata language from the metadata file.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with language detection results
    """
    if state.get('final_header') is None:
        error_msg = "Cannot detect header language: no header available"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    logger.info("Detecting language of headers and checking metadata")
    
    try:
        header = state['final_header']
        
        # Join headers into a single string for language detection
        header_text = ' '.join([str(h) for h in header if h])
        
        # Detect the language of the header
        header_language, header_confidence = detect_language(header_text)
        normalized_header_lang = normalize_language_name(header_language)
        
        # Get the language from metadata file
        metadata_language, metadata_confidence = get_metadata_language(state)
        normalized_metadata_lang = normalize_language_name(metadata_language)
        
        logger.info(f"Detected header language: {header_language} (confidence: {header_confidence:.2f})")
        logger.info(f"Metadata language: {metadata_language} (confidence: {metadata_confidence:.2f})")
        
        # Check if the header language matches the metadata language
        is_same_language = are_languages_matching(header_language, metadata_language)
        
        # Update the state with language detection results
        state['header_language'] = header_language
        state['header_language_confidence'] = header_confidence
        state['metadata_language'] = metadata_language
        state['metadata_language_confidence'] = metadata_confidence
        state['is_header_in_target_language'] = is_same_language
        
        if not is_same_language:
            logger.info(f"Header language ({header_language}) does not match metadata language ({metadata_language}), translation needed")
        else:
            logger.info(f"Header language already matches metadata language ({metadata_language}), no translation needed")
        
    except Exception as e:
        error_msg = f"Error detecting header language: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        # Default to requiring translation if we can't detect the language
        state['is_header_in_target_language'] = False
    
    return state
