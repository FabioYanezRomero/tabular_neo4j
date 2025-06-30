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

def perform_column_analytics_node(state: GraphState, node_order: int) -> GraphState:
    """
    Perform statistical and pattern analysis on each column.
    
    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline
        
    Returns:
        Updated graph state with column_analytics
    """
    if state.get('processed_dataframe') is None:
        error_msg = "Cannot analyze columns: no processed dataframe available"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        # Ensure the returned state is always a GraphState instance
        if not isinstance(state, GraphState):
            state = GraphState.from_dict(dict(state))
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
        
        # Perform comprehensive rule-based classification for each column
        rule_based_classification = {}
        df = state['processed_dataframe']
        row_count = len(df) if df is not None else 0
        
        for column_name, analytics in analytics_results.items():
            # Extract analytics metrics
            uniqueness = analytics.get('uniqueness', 0)  # Ratio of unique values
            cardinality = analytics.get('cardinality', 0)  # Number of unique values
            data_type = analytics.get('data_type', 'unknown')
            missing_percentage = analytics.get('missing_percentage', 0)  # Ratio of missing values
            value_lengths = analytics.get('value_lengths', {})  # Statistics about value lengths
            avg_value_length = value_lengths.get('mean', 0) if value_lengths else 0
            max_value_length = value_lengths.get('max', 0) if value_lengths else 0
            
            # Calculate additional metrics
            missing_ratio = missing_percentage  # Already between 0 and 1
            value_diversity = cardinality / row_count if row_count > 0 else 0  # Normalized cardinality
            
            # Get patterns from analytics data
            patterns = analytics.get('patterns', {})

            # Classification based only on data_type and patterns
            if data_type in ['date', 'datetime', 'float', 'integer']:
                classification = "property"
                confidence = 1.0
                reason = f"forced property classification due to {data_type} data type"
            elif 'email' in patterns and patterns['email'] > 0.5:
                classification = "property"
                confidence = 1.0
                reason = f"forced property classification due to email pattern"
            elif any(p in patterns for p in ['phone', 'credit_card', 'numeric_id']) and patterns.get(next((p for p in patterns if p in ['phone', 'credit_card', 'numeric_id']), None), 0) > 0.5:
                classification = "property"
                confidence = 1.0
                reason = f"forced property classification due to numeric pattern"
            else:
                classification = "entity"
                confidence = 1.0
                reason = "classified as entity by default (no property pattern detected)"

            # Log detailed analysis
            logger.debug(f"Column '{column_name}' analysis: uniqueness={uniqueness:.2f}, cardinality={cardinality}, "
                        f"type={data_type}, missing={missing_ratio:.2f}, avg_length={avg_value_length:.1f}")
            logger.debug(f"Raw metrics: uniqueness={uniqueness}, cardinality={cardinality}, data_type={data_type}, missing_percentage={missing_percentage}, avg_value_length={avg_value_length}, max_value_length={max_value_length}")
            logger.debug(f"Classification: {classification} (confidence: {confidence:.2f})")

            # Store the rule-based classification with detailed information (raw metrics as 'scores')
            rule_based_classification[column_name] = {
                'column_name': column_name,
                'classification': classification,
                'confidence': confidence,
                'reason': reason,
                'scores': {
                    'uniqueness': uniqueness,
                    'cardinality': cardinality,
                    'data_type': data_type,
                    'missing_percentage': missing_percentage,
                    'avg_value_length': avg_value_length,
                    'max_value_length': max_value_length,
                },
                'analytics': analytics
            }
        
        # Update the state with the rule-based classification
        state['rule_based_classification'] = rule_based_classification
        logger.info(f"Completed rule-based classification for {len(rule_based_classification)} columns")
        
    except Exception as e:
        error_msg = f"Error analyzing columns: {str(e)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
    
    # Ensure the returned state is always a GraphState instance
    if not isinstance(state, GraphState):
        state = GraphState.from_dict(dict(state))
    return state
