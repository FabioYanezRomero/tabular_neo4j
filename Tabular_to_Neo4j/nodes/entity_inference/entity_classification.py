"""
Entity classification module for schema synthesis in the Tabular to Neo4j converter.
This module handles the first step in schema synthesis: classifying columns as entities or properties.
"""

from typing import Dict, Any
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.csv_utils import get_primary_entity_from_filename
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
    if state.get('llm_column_semantics') is None:
        missing_inputs.append("llm_column_semantics")
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
    semantics_count = len(state.get('llm_column_semantics', {}))
    logger.debug(f"Input data: {header_count} columns, {analytics_count} analytics entries, {semantics_count} semantic entries")

    try:
        logger.debug("Initializing classification process")
        # Initialize classification dictionary
        classification = {}

        # Get primary entity label from filename
        csv_path = state['csv_file_path']
        primary_entity_label = get_primary_entity_from_filename(csv_path)
        file_name = os.path.basename(csv_path)
        logger.info(f"Derived primary entity label '{primary_entity_label}' from file: {file_name}")

        # Process each column using LLM for classification
        logger.info(f"Beginning classification of {len(state['final_header'])} columns")
        processed_columns = 0
        skipped_columns = 0

        for column_name in state['final_header']:
            logger.debug(f"Processing column: '{column_name}'")

            # Get analytics and semantics for this column
            analytics = state.get('column_analytics', {}).get(column_name, {})
            semantics = state.get('llm_column_semantics', {}).get(column_name, {})

            # Skip if we don't have both analytics and semantics
            if not analytics:
                logger.warning(f"Missing analytics for column '{column_name}', skipping")
                skipped_columns += 1
                continue
            if not semantics:
                logger.warning(f"Missing semantics for column '{column_name}', skipping")
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

            # Format the prompt with column information and metadata
            prompt = format_prompt('classify_entities_properties.txt',
                                  file_name=file_name,
                                  column_name=column_name,
                                  primary_entity=primary_entity_label,
                                  sample_values=str(sample_values),
                                  uniqueness_ratio=analytics.get('uniqueness_ratio', 0),
                                  cardinality=analytics.get('cardinality', 0),
                                  data_type=analytics.get('data_type', 'unknown'),
                                  missing_percentage=analytics.get('missing_percentage', 0) * 100,
                                  semantic_type=semantics.get('semantic_type', 'Unknown'),
                                  llm_role=semantics.get('neo4j_role', 'UNKNOWN'),
                                  metadata_text=metadata_text)

            # Call the LLM for entity/property classification
            logger.debug(f"Calling LLM for entity/property classification of column '{column_name}'")
            try:
                response = call_llm_with_json_output(prompt, state_name="classify_entities_properties")
                logger.debug(f"Received LLM classification response for '{column_name}'")

                # Extract the classification results
                classification_result = response.get('classification', 'entity_property')
                entity_type = response.get('entity_type', primary_entity_label)
                relationship = response.get('relationship_to_primary', '')

                classification[column_name] = {
                    'column_name': column_name,
                    'classification': classification_result,
                    'entity_type': entity_type,
                    'relationship_to_primary': relationship,
                    'property_name': response.get('property_name', column_name),
                    'reasoning': response.get('reasoning', ''),
                    'analytics': analytics,
                    'semantics': semantics
                }

                logger.info(f"Classified '{column_name}' as '{classification_result}' with entity type '{entity_type}'")
                if relationship:
                    logger.debug(f"Relationship to primary entity: '{relationship}'")

            # TODO: check and improve the fallback evaluation whenever the model fails
            except Exception as e:
                logger.error(f"LLM entity classification failed for '{column_name}': {str(e)}")

                # If LLM call fails, use a fallback classification based on analytics
                uniqueness = analytics.get('uniqueness', 0)
                is_unique_enough = uniqueness > UNIQUENESS_THRESHOLD
                fallback_classification = 'entity' if is_unique_enough else 'property'

                logger.warning(f"Using fallback classification for '{column_name}': {fallback_classification} based on uniqueness ({uniqueness:.2f})")

                classification[column_name] = {
                    'column_name': column_name,
                    'classification': fallback_classification,
                    'entity_type': primary_entity_label if is_unique_enough else '',
                    'relationship_to_primary': 'IS_A' if is_unique_enough else '',
                    'property_name': column_name,
                    'reasoning': f'Fallback classification based on uniqueness ({uniqueness:.2f})',
                    'analytics': analytics,
                    'semantics': semantics
                }

                state['error_messages'].append(f"LLM entity classification failed for {column_name}: {str(e)}")

            # If we're in the exception handler, we've already set the fallback classification
            # No need to do further classification based on neo4j_role
            pass

        # Update the state
        state['entity_property_classification'] = classification

    except Exception as e:
        logger.error(f"Error classifying entities/properties: {str(e)}")
        state['error_messages'].append(f"Error classifying entities/properties: {str(e)}")
        state['entity_property_classification'] = {}

    return state
