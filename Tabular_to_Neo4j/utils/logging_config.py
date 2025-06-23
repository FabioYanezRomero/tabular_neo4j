"""
Centralized logging configuration for the Tabular to Neo4j converter.
This module provides a consistent logging setup across the application.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

# Default log levels for different components
DEFAULT_LOG_LEVELS = {
    "root": logging.INFO,
    "Tabular_to_Neo4j": logging.INFO,
    "Tabular_to_Neo4j.nodes": logging.INFO,
    "Tabular_to_Neo4j.utils": logging.INFO,
    "Tabular_to_Neo4j.nodes.input_nodes": logging.INFO,
    "Tabular_to_Neo4j.nodes.header_nodes": logging.INFO,
    "Tabular_to_Neo4j.nodes.analysis_nodes": logging.INFO,
    "Tabular_to_Neo4j.nodes.entity_inference": logging.INFO,
    "Tabular_to_Neo4j.nodes.db_schema": logging.INFO,
    "Tabular_to_Neo4j.utils.csv_utils": logging.INFO,
    "Tabular_to_Neo4j.utils.llm_manager": logging.INFO,
    "Tabular_to_Neo4j.utils.language_utils": logging.INFO,
    "langgraph": logging.WARNING,
    "langchain": logging.WARNING,
}

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

_LOG_FILE_HANDLER = None

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    detailed_format: bool = False,
    component_levels: Optional[Dict[str, int]] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Overall log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to stdout only)
        detailed_format: Whether to use detailed log format with filename and line number
        component_levels: Dictionary of component-specific log levels
    """
    # Convert string log level to logging constant if provided
    root_level = logging.INFO
    if log_level:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        root_level = level_map.get(log_level.upper(), logging.INFO)
    
    # Create handlers
    handlers = []
    log_format = DETAILED_LOG_FORMAT if detailed_format else LOG_FORMAT
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        global _LOG_FILE_HANDLER
        _LOG_FILE_HANDLER = logging.FileHandler(log_file)
        _LOG_FILE_HANDLER.setFormatter(logging.Formatter(log_format))
        handlers.append(_LOG_FILE_HANDLER)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        handlers.append(handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set component-specific log levels
    levels = DEFAULT_LOG_LEVELS.copy()
    if component_levels:
        levels.update(component_levels)
    
    for logger_name, level in levels.items():
        logging.getLogger(logger_name).setLevel(level)
    
    # Log startup information
    root_logger.info(f"Logging initialized with root level: {logging.getLevelName(root_level)}")
    if detailed_format:
        root_logger.info("Using detailed log format with file and line information")
    if log_file:
        root_logger.info(f"Logging to file: {log_file}")

def set_log_file_path(log_file_path: str, detailed_format: bool = False):
    """
    Dynamically update the log file handler to write logs to a new file (e.g., after OutputSaver is initialized).
    """
    global _LOG_FILE_HANDLER
    log_format = DETAILED_LOG_FORMAT if detailed_format else LOG_FORMAT
    root_logger = logging.getLogger()
    if _LOG_FILE_HANDLER:
        root_logger.removeHandler(_LOG_FILE_HANDLER)
    _LOG_FILE_HANDLER = logging.FileHandler(log_file_path)
    _LOG_FILE_HANDLER.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(_LOG_FILE_HANDLER)
    root_logger.info(f"Log file handler updated to: {log_file_path}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
