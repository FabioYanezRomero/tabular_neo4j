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
            
            # Initialize score components (higher = more likely to be an entity)
            uniqueness_score = 0
            cardinality_score = 0
            data_type_score = 0
            missing_value_score = 0
            value_length_score = 0
            
            # 1. Uniqueness Analysis (0-5 points)
            # High uniqueness suggests entity
            if uniqueness > 0.9:  # Almost all values are unique
                uniqueness_score = 5
            elif uniqueness > 0.7:  # Many values are unique
                uniqueness_score = 4
            elif uniqueness > 0.5:  # Half the values are unique
                uniqueness_score = 3
            elif uniqueness > 0.3:  # Some values are unique
                uniqueness_score = 2
            elif uniqueness > 0.1:  # Few values are unique
                uniqueness_score = 1
            
            # 2. Cardinality Analysis (0-4 points)
            # High cardinality relative to row count suggests entity
            if value_diversity > 0.8:  # Very high diversity of values
                cardinality_score = 4
            elif value_diversity > 0.5:  # High diversity
                cardinality_score = 3
            elif value_diversity > 0.2:  # Moderate diversity
                cardinality_score = 2
            elif value_diversity > 0.05:  # Low diversity but still significant
                cardinality_score = 1
            
            # 3. Data Type Analysis (0-3 points)
            # Text and categorical data more likely to be entities
            if data_type in ['string', 'text', 'object']:
                if avg_value_length > 10:  # Longer text values suggest entities (names, descriptions)
                    data_type_score = 3
                elif avg_value_length > 5:  # Medium length text
                    data_type_score = 2
                else:  # Short text could be codes or abbreviations
                    data_type_score = 1
            elif data_type in ['categorical']:
                data_type_score = 2  # Categorical data could be entities
            elif data_type in ['date', 'datetime']:  # Dates rarely represent entities
                data_type_score = 0
            elif data_type in ['integer', 'float']:  # Numeric data less likely to be entities
                # Exception: if high uniqueness, could be IDs
                if uniqueness > 0.9 and 'id' in column_name.lower():
                    data_type_score = 2  # Likely numeric IDs
                else:
                    data_type_score = 0
            
            # 4. Missing Value Analysis (0-2 points)
            # Entities tend to have fewer missing values
            if missing_ratio < 0.05:  # Very few missing values
                missing_value_score = 2
            elif missing_ratio < 0.2:  # Some missing values
                missing_value_score = 1
            
            # 5. Value Length Analysis (0-2 points)
            # Entities often have consistent, non-trivial length
            if avg_value_length > 3 and max_value_length < 100:  # Reasonable length for entities
                value_length_score = 2
            elif avg_value_length > 1:  # Short but not trivial
                value_length_score = 1
            
            # Calculate total score (maximum possible: 16 points)
            total_score = uniqueness_score + cardinality_score + data_type_score + missing_value_score + value_length_score
            
            # Determine classification based on total score
            # Higher threshold for entity classification (> 60% of max score)
            entity_threshold = 9  # ~60% of maximum 16 points
            
            if total_score >= entity_threshold:
                classification = "entity"
                # Confidence based on how far above threshold
                confidence = 0.7 + min(0.3, (total_score - entity_threshold) / 10)
                reason = f"entity score: {total_score}/{16}"
            else:
                classification = "property"
                # Confidence based on how far below threshold
                confidence = 0.7 + min(0.3, (entity_threshold - total_score) / 10)
                reason = f"property score: {total_score}/{16}"
            
            # Log detailed analysis
            logger.debug(f"Column '{column_name}' analysis: uniqueness={uniqueness:.2f}, cardinality={cardinality}, "
                        f"type={data_type}, missing={missing_ratio:.2f}, avg_length={avg_value_length:.1f}")
            logger.debug(f"Scores: uniqueness={uniqueness_score}, cardinality={cardinality_score}, "
                        f"data_type={data_type_score}, missing={missing_value_score}, value_length={value_length_score}")
            logger.debug(f"Classification: {classification} (confidence: {confidence:.2f}, total score: {total_score})")
            
            logger.debug(f"Enhanced rule-based classification: '{column_name}' is a {classification} (reason: {reason}, confidence: {confidence:.2f})")
            
            # Store the rule-based classification with detailed information
            rule_based_classification[column_name] = {
                'column_name': column_name,
                'classification': classification,
                'confidence': confidence,
                'reason': reason,
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
