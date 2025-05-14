"""
Entity classification module for schema synthesis in the Tabular to Neo4j converter.
This module handles the first step in schema synthesis: classifying columns as entities or properties.
"""

from typing import Dict, Any
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.llm_manager import format_prompt, call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_for_state, format_metadata_for_prompt
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def classify_entities_properties_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """First step in schema synthesis: Classify columns as entities or properties based on analytics and semantics.
    Uses LLM to classify each column based on its analytics and semantics.

    Args:
        state: The current graph state
        config: LangGraph runnable configuration

    Returns:
        Updated graph state with entity_property_classification
    """
    logger.info("Starting entity-property classification process")

    # Validate required inputs
    missing_inputs = []
    if state.get('column_analytics') is None:
        missing_inputs.append("column_analytics")
    if state.get('final_header') is None:
        missing_inputs.append("final_header")

    if missing_inputs:
        error_msg = f"Cannot classify entities/properties: missing required analysis data: {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state['error_messages'].append(error_msg)
        return state

    # Log input data stats
    header_count = len(state['final_header'])
    analytics_count = len(state.get('column_analytics', {}))
    logger.debug(f"Input data: {header_count} columns, {analytics_count} analytics entries")

    try:
        logger.debug("Initializing classification process")
        # Initialize classification dictionary
        classification = {}
        
        
        # Process each column using LLM for classification
        logger.info(f"Beginning classification of {len(state['final_header'])} columns")
        processed_columns = 0
        skipped_columns = 0
        
        # Get rule-based classification if available
        rule_based = state.get('rule_based_classification', {})

        try:
            for column_name in state['final_header']:
                try:
                    logger.debug(f"Processing column: '{column_name}'")
    
                    # Get analytics for this column
                    analytics = state.get('column_analytics', {}).get(column_name, {})
    
                    # Skip if we don't have analytics
                    if not analytics:
                        logger.warning(f"Missing analytics for column '{column_name}', skipping")
                        skipped_columns += 1
                        continue
    
                    # Log key analytics for debugging
                    uniqueness = analytics.get('uniqueness', 0)
                    null_percentage = analytics.get('null_percentage', 0)
                    logger.debug(f"Column '{column_name}' analytics: uniqueness={uniqueness:.2f}, null_percentage={null_percentage:.2f}")
    
                    # Get sample values for this column
                    sample_values = []
                    if state.get('processed_dataframe') is not None:
                        df = state['processed_dataframe']
                        non_null_values = df[column_name].dropna()
    
                        if len(non_null_values) > 0:
                            sample_count = min(5, len(non_null_values))
                            sample_values = non_null_values.sample(sample_count).tolist()
                            logger.debug(f"Sampled {len(sample_values)} values from column '{column_name}'")
                        else:
                            logger.warning(f"Column '{column_name}' has no non-null values to sample")
                    else:
                        logger.warning("No processed dataframe available for sampling values")
    
                    processed_columns += 1
    
                    # Get metadata for the CSV file
                    metadata = get_metadata_for_state(state)
                    metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
    
                    # Check if we have a rule-based classification for this column
                    rule_based_class = rule_based.get(column_name, {}).get('classification', None)
                    
                    # If we have a rule-based classification with high confidence, use it directly
                    if rule_based_class and rule_based.get(column_name, {}).get('confidence', 0) > 0.8:
                        logger.info(f"Using rule-based classification for '{column_name}': {rule_based_class}")
                        classification[column_name] = {
                            'column_name': column_name,
                            'classification': rule_based_class,
                            'confidence': rule_based.get(column_name, {}).get('confidence', 0.9),
                            'analytics': analytics,
                            'source': 'rule_based'
                        }
                        continue
    
                    # Format the prompt with column information and metadata
                    prompt = format_prompt('classify_entities_properties.txt',
                                          column_name=column_name,
                                          sample_values=str(sample_values),
                                          uniqueness_ratio=analytics.get('uniqueness', 0),
                                          cardinality=analytics.get('cardinality', 0),
                                          data_type=analytics.get('data_type', 'unknown'),
                                          missing_percentage=analytics.get('null_percentage', 0) * 100,
                                          metadata_text=metadata_text)
    
                    # Call the LLM for entity/property classification
                    logger.debug(f"Calling LLM for entity/property classification of column '{column_name}'")
                    try:
                        # Add a fallback classification based on analytics
                        fallback_classification = 'entity' if uniqueness > UNIQUENESS_THRESHOLD else 'property'
                        
                        # Call the LLM for classification
                        response = call_llm_with_json_output(prompt, state_name="classify_entities_properties")
                        logger.debug(f"Received LLM classification response for '{column_name}': {response}")
                        
                        # If response is empty or has an error, use fallback
                        if not response or 'error' in response:
                            logger.warning(f"Using fallback classification for '{column_name}' due to LLM error")
                            response = {
                                'classification': fallback_classification,
                                'confidence': 0.6,
                                'reasoning': 'Fallback classification based on uniqueness threshold.'
                            }
    
                        # Extract the classification results
                        classification_result = response.get('classification', 'entity_property')
                        # Handle the case where classification_result is not a string
                        if not isinstance(classification_result, str):
                            if isinstance(classification_result, dict) and 'type' in classification_result:
                                classification_result = classification_result['type']
                            else:
                                classification_result = 'entity_property'  # Default fallback
                        
                        # Normalize the classification result
                        classification_result = classification_result.lower().strip()
                        if classification_result not in ['entity', 'property', 'entity_property']:
                            if 'entity' in classification_result:
                                classification_result = 'entity'
                            elif 'property' in classification_result:
                                classification_result = 'property'
                            else:
                                classification_result = 'entity_property'
                        
                        # Handle confidence value
                        confidence = response.get('confidence', 0.0)
                        if not isinstance(confidence, (int, float)):
                            try:
                                confidence = float(confidence)
                            except (ValueError, TypeError):
                                confidence = 0.5  # Default confidence
    
                        classification[column_name] = {
                            'column_name': column_name,
                            'classification': classification_result,
                            'confidence': confidence,
                            'analytics': analytics,
                            'source': 'llm'
                        }
                        
                        logger.info(f"Classified '{column_name}' as '{classification_result}' with confidence '{confidence}' using LLM approach")
                    except Exception as e:
                        # If LLM classification fails, use fallback based on uniqueness
                        fallback_classification = 'entity' if uniqueness > UNIQUENESS_THRESHOLD else 'property'
                        logger.warning(f"LLM classification failed for '{column_name}', using fallback: {fallback_classification}")
                        classification[column_name] = {
                            'column_name': column_name,
                            'classification': fallback_classification,
                            'confidence': 0.6,  # Moderate confidence for fallback
                            'analytics': analytics,
                            'source': 'fallback'
                        }
                except Exception as column_error:
                    # If processing a specific column fails, log it and continue with the next one
                    logger.error(f"Error processing column '{column_name}': {str(column_error)}")
                    # Use a default classification to ensure the pipeline continues
                    classification[column_name] = {
                        'column_name': column_name,
                        'classification': 'error',  # Default to property as safer option
                        'confidence': 0.5,
                        'analytics': analytics if 'analytics' in locals() else {},
                        'source': 'error_fallback'
                    }
            
            # If we processed at least one column successfully, consider it a success
            if processed_columns > 0:
                logger.info(f"Successfully classified {processed_columns} columns, skipped {skipped_columns}")
            else:
                logger.warning("No columns were successfully classified")
                
        except Exception as e:
            logger.error(f"Error in classification process: {str(e)}")
            # Create a minimal classification for each column to allow the pipeline to continue
            for column_name in state['final_header']:
                if column_name not in classification:
                    # Use rule-based classification if available, otherwise default to property
                    if column_name in rule_based:
                        classification[column_name] = rule_based[column_name]
                    else:
                        classification[column_name] = {
                            'column_name': column_name,
                            'classification': 'error',  # Default to property as safer option
                            'confidence': 0.5,
                            'analytics': state.get('column_analytics', {}).get(column_name, {}),
                            'source': 'global_error_fallback'
                        }


        # Update the state
        state['entity_property_classification'] = classification

    except Exception as e:
        logger.error(f"Error classifying entities/properties: {str(e)}")
        state['error_messages'].append(f"Error classifying entities/properties: {str(e)}")
        state['entity_property_classification'] = {}

    return state
