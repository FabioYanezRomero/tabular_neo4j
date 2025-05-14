"""
Column analytics module for the Tabular to Neo4j converter.
This module handles statistical and pattern analysis of columns.
"""

from typing import Dict, Any, List
import pandas as pd
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.analytics_utils import analyze_all_columns
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def perform_column_analytics_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Perform statistical and pattern analysis on each column.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with column_analytics
    """
    if state.get('processed_dataframe') is None:
        error_msg = "Cannot analyze columns: no processed dataframe available"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    logger.info("Performing statistical and pattern analysis on columns")
    
    try:
        # Analyze all columns in the processed dataframe
        analytics_results = analyze_all_columns(state['processed_dataframe'])
        
        # Log the results
        logger.info(f"Successfully analyzed {len(analytics_results)} columns")
        
        # Add detailed logs for each column
        for column_name, analytics in analytics_results.items():
            data_type = analytics.get('data_type', 'unknown')
            uniqueness = analytics.get('uniqueness', 0)
            cardinality = analytics.get('cardinality', 0)
            missing_percentage = analytics.get('missing_percentage', 0) * 100
            
            logger.debug(f"Column '{column_name}': type={data_type}, uniqueness={uniqueness:.2f}, "
                        f"cardinality={cardinality}, missing={missing_percentage:.2f}%")
        
        # Update the state with the analytics results
        state['column_analytics'] = analytics_results
        
        # Perform rule-based classification for each column
        rule_based_classification = {}
        df = state['processed_dataframe']
        
        for column_name, analytics in analytics_results.items():
            # Determine rule-based classification
            uniqueness = analytics.get('uniqueness', 0)
            cardinality = analytics.get('cardinality', 0)
            
            if uniqueness > UNIQUENESS_THRESHOLD:
                classification = "entity"
                logger.debug(f"Rule-based classification: '{column_name}' is an entity (uniqueness: {uniqueness:.2f})")
            elif cardinality < len(df) * 0.1 and cardinality > 1:
                classification = "entity"
                logger.debug(f"Rule-based classification: '{column_name}' is an entity (low cardinality: {cardinality})")
            else:
                classification = "property"
                logger.debug(f"Rule-based classification: '{column_name}' is a property")
            
            # Store the rule-based classification
            rule_based_classification[column_name] = {
                'column_name': column_name,
                'classification': classification,
                'confidence': 0.8 if uniqueness > UNIQUENESS_THRESHOLD else 0.6,  # Higher confidence for uniqueness-based classification
                'analytics': analytics
            }
        
        # Update the state with the rule-based classification
        state['rule_based_classification'] = rule_based_classification
        logger.info(f"Completed rule-based classification for {len(rule_based_classification)} columns")
        
    except Exception as e:
        error_msg = f"Error analyzing columns: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    return state
