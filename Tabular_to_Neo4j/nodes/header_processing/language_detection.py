"""
Language detection module for the Tabular to Neo4j converter.
This module handles detecting the language of headers.
"""

from typing import Dict, Any, List
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.language_utils import verify_header_language, detect_language, normalize_language_name
from Tabular_to_Neo4j.config import TARGET_HEADER_LANGUAGE
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def detect_header_language_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Detect the language of headers and determine if translation is needed.
    This node checks if the header language matches the target language in METADATA.
    
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
    
    logger.info("Detecting language of headers")
    
    try:
        header = state['final_header']
        
        # Join headers into a single string for language detection
        header_text = ' '.join([str(h) for h in header if h])
        
        # Detect the language of the header
        detected_language, confidence = detect_language(header_text)
        normalized_detected = normalize_language_name(detected_language)
        normalized_target = normalize_language_name(TARGET_HEADER_LANGUAGE)
        
        logger.info(f"Detected header language: {detected_language} (confidence: {confidence:.2f})")
        logger.info(f"Target language: {TARGET_HEADER_LANGUAGE}")
        
        # Check if the detected language matches the target language
        is_target_language = verify_header_language(header_text, TARGET_HEADER_LANGUAGE)
        
        # Update the state with language detection results
        state['header_language'] = detected_language
        state['header_language_confidence'] = confidence
        state['is_header_in_target_language'] = is_target_language
        
        if not is_target_language:
            logger.info(f"Header language ({detected_language}) does not match target language ({TARGET_HEADER_LANGUAGE}), translation needed")
        else:
            logger.info(f"Header language already matches target language ({TARGET_HEADER_LANGUAGE}), no translation needed")
        
    except Exception as e:
        error_msg = f"Error detecting header language: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        # Default to requiring translation if we can't detect the language
        state['is_header_in_target_language'] = False
    
    return state
