"""
Entity reconciliation module for entity inference in the Tabular to Neo4j converter.
This module handles reconciling different classification approaches.
"""

from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def reconcile_entity_property_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Reconcile analytics-based and LLM-based classifications
    to create a consensus model of entities and properties.
    Uses LLM to reconcile different classification approaches.
    
    Args:
        state: The current graph state
        config: LangGraph runnable configuration
        
    Returns:
        Updated graph state with entity_property_consensus
    """
    logger.info("Starting entity-property reconciliation process")
    
    # Validate required inputs
    if state.get('entity_property_classification') is None:
        error_msg = "Cannot reconcile entity/property classifications: missing entity_property_classification"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state
    
    # Debug: Log state keys to understand what's available
    logger.info(f"State keys: {list(state.keys())}")
    
    # Check if rule-based classification is available
    has_rule_based = 'rule_based_classification' in state and state['rule_based_classification']
    if has_rule_based:
        logger.info(f"Rule-based classification exists with {len(state['rule_based_classification'])} entries")
    else:
        logger.info("Rule-based classification is missing or empty - will use only LLM classification")
    
    try:
        # Get the LLM classification
        llm_classification = state['entity_property_classification']
        
        # Create rule-based classification directly from analytics if not in state
        if 'rule_based_classification' not in state or not state['rule_based_classification']:
            logger.info("Creating rule-based classification from analytics")
            rule_based_classification = {}
            
            for column_name, analytics_data in state.get('column_analytics', {}).items():
                uniqueness = analytics_data.get('uniqueness', 0)
                cardinality = analytics_data.get('cardinality', 0)
                
                if uniqueness > UNIQUENESS_THRESHOLD:
                    classification = "entity"
                    confidence = 0.8
                elif cardinality < len(state.get('processed_dataframe', [])) * 0.1 and cardinality > 1:
                    classification = "entity"
                    confidence = 0.7
                else:
                    classification = "property"
                    confidence = 0.6
                    
                rule_based_classification[column_name] = {
                    'column_name': column_name,
                    'classification': classification,
                    'confidence': confidence,
                    'analytics': analytics_data,
                    'source': 'rule_based'
                }
            
            logger.info(f"Created rule-based classification for {len(rule_based_classification)} columns")
        else:
            rule_based_classification = state['rule_based_classification']
        
        # Initialize consensus dictionary
        consensus = {}
        
        # Process each column to determine if reconciliation is needed
        for column_name in state.get('final_header', []):
            # Get both classifications for this column
            llm_info = llm_classification.get(column_name, {})
            rule_info = rule_based_classification.get(column_name, {})
            
            # Skip if we don't have both classifications
            if not llm_info and not rule_info:
                logger.warning(f"Missing classification for column '{column_name}', using fallback classification")
                
                # Use a fallback classification based on column name patterns
                fallback_classification = 'entity' if 'id' in column_name.lower() or 'name' in column_name.lower() else 'property'
                consensus[column_name] = {'column_name': column_name, 'classification': fallback_classification, 'confidence': 0.5}
                continue
            elif not llm_info:
                logger.warning(f"Missing LLM classification for column '{column_name}', using rule-based classification")
                consensus[column_name] = rule_info
                continue
            elif not rule_info:
                logger.warning(f"Missing rule-based classification for column '{column_name}', using LLM classification")
                consensus[column_name] = llm_info
                continue
            
            # Check if there's a discrepancy between the classifications
            llm_classification_result = llm_info.get('classification', '')
            rule_classification_result = rule_info.get('classification', '')
            
            # If classifications match, no need for reconciliation
            if llm_classification_result == rule_classification_result:
                logger.info(f"Classifications match for column '{column_name}': {llm_classification_result}")
                
                # Use the LLM classification as it typically has more detailed reasoning
                consensus[column_name] = llm_info
                continue
            
            # If there's a discrepancy, perform reconciliation
            logger.info(f"Classification discrepancy for '{column_name}': LLM={llm_classification_result}, Rule={rule_classification_result}")
            
            # Reconciliation strategy: 
            # 1. Compare confidence scores
            # 2. If one is significantly higher, use that classification
            # 3. Otherwise, prefer LLM classification as it's more contextual
            llm_confidence = llm_info.get('confidence', 0.5)
            rule_confidence = rule_info.get('confidence', 0.5)
            
            if llm_confidence >= rule_confidence + 0.2:  # LLM is significantly more confident
                logger.info(f"Using LLM classification for '{column_name}' due to higher confidence: {llm_confidence} vs {rule_confidence}")
                consensus[column_name] = llm_info
            elif rule_confidence >= llm_confidence + 0.2:  # Rule-based is significantly more confident
                logger.info(f"Using rule-based classification for '{column_name}' due to higher confidence: {rule_confidence} vs {llm_confidence}")
                consensus[column_name] = rule_info
            else:  # Similar confidence levels, prefer LLM
                logger.info(f"Using LLM classification for '{column_name}' as default reconciliation strategy")
                consensus[column_name] = llm_info
        
        # Add uniqueness information for entities to help with later processing
        for column_name, info in consensus.items():
            if info['classification'] == 'entity':
                analytics = state.get('column_analytics', {}).get(column_name, {})
                info['uniqueness_ratio'] = analytics.get('uniqueness_ratio', 0)
        
        # Update the state
        state['entity_property_consensus'] = consensus
        
    except Exception as e:
        logger.error(f"Error reconciling entity/property classifications: {str(e)}")
        state['error_messages'].append(f"Error reconciling entity/property classifications: {str(e)}")
        state['entity_property_consensus'] = state.get('entity_property_classification', {})
    
    return state
