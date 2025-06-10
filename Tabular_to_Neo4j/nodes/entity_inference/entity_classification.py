"""
Entity classification module for schema synthesis in the Tabular to Neo4j converter.
This module handles the first step in schema synthesis: classifying columns as entities or properties.
"""

from typing import Dict, Any
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import (
    get_metadata_for_state,
    format_metadata_for_prompt,
)
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def classify_entities_properties_node(
    state: GraphState, config: RunnableConfig
) -> GraphState:
    """First step in schema synthesis: Classify columns as entities or properties using both LLM and rule-based approaches.

    Args:
        state: The current graph state
        config: LangGraph runnable configuration

    Returns:
        Updated graph state with entity_property_classification and rule_based_classification
    """
    logger.info("Starting entity-property classification process")

    # Validate required inputs
    if state.get("column_analytics") is None or state.get("final_header") is None:
        error_msg = "Cannot classify entities/properties: missing required data"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        return state

    try:
        # Get the column analytics and header
        analytics = state["column_analytics"]
        header = state["final_header"]

        # Get the rule-based classification from column_analytics
        rule_based_classification = state.get("rule_based_classification", {})

        # Initialize LLM classification dictionary
        llm_classification = {}

        # Process each column using both LLM and rule-based approaches
        logger.info(f"Beginning classification of {len(header)} columns")
        processed_columns = 0
        skipped_columns = 0

        try:
            for column_name in state["final_header"]:
                try:
                    logger.debug(f"Processing column: '{column_name}'")

                    # Get analytics for this column
                    analytics = state.get("column_analytics", {}).get(column_name, {})

                    # Skip if we don't have analytics
                    if not analytics:
                        logger.warning(
                            f"Missing analytics for column '{column_name}', skipping"
                        )
                        skipped_columns += 1
                        continue

                    # Log key analytics for debugging
                    uniqueness = analytics.get("uniqueness", 0)
                    null_percentage = analytics.get("null_percentage", 0)
                    logger.debug(
                        f"Column '{column_name}' analytics: uniqueness={uniqueness:.2f}, null_percentage={null_percentage:.2f}"
                    )

                    # Get full sample rows for context and sample values for this column
                    sample_values_json = "[]"
                    full_sample_json = "[]"
                    if state.get("processed_dataframe") is not None:
                        df = state["processed_dataframe"]

                        # Use the existing get_sample_rows and df_to_json_sample functions
                        from Tabular_to_Neo4j.utils.csv_utils import (
                            get_sample_rows,
                            df_to_json_sample,
                        )

                        # Get sample rows (up to 5) from the entire dataframe
                        sample_count = 5
                        full_sample_df = get_sample_rows(df, sample_count)

                        if not full_sample_df.empty:
                            # Get the full sample data as a JSON string
                            full_sample_json = df_to_json_sample(full_sample_df)

                            # Extract just the values for this column as a list
                            sample_values = full_sample_df[column_name].tolist()

                            # Convert to JSON string
                            import json

                            sample_values_json = json.dumps(sample_values)
                            logger.debug(
                                f"Sampled {len(sample_values)} values from column '{column_name}'"
                            )
                        else:
                            logger.warning(f"No sample data available")
                    else:
                        logger.warning(
                            "No processed dataframe available for sampling values"
                        )

                    processed_columns += 1

                    # Get metadata for the CSV file
                    metadata = get_metadata_for_state(state)
                    metadata_text = (
                        format_metadata_for_prompt(metadata)
                        if metadata
                        else "No metadata available."
                    )

                    # Check if we have a rule-based classification for this column from the state
                    rule_based = state.get("rule_based_classification", {})
                    rule_based_class = rule_based.get(column_name, {}).get(
                        "classification", None
                    )

                    # If we have a rule-based classification with high confidence, use it directly
                    if (
                        rule_based_class
                        and rule_based.get(column_name, {}).get("confidence", 0) > 0.8
                    ):
                        logger.info(
                            f"Using rule-based classification for '{column_name}': {rule_based_class}"
                        )
                        llm_classification[column_name] = {
                            "column_name": column_name,
                            "classification": rule_based_class,
                            "confidence": rule_based.get(column_name, {}).get(
                                "confidence", 0.9
                            ),
                            "analytics": analytics,
                            "source": "rule_based",
                        }
                        continue

                    # Format the prompt with column information and metadata
                    try:
                        prompt = format_prompt(
                            "classify_entities_properties_v3.txt",
                            column_name=column_name,
                            full_sample_data=full_sample_json,  # Include the full sample data
                            uniqueness_ratio=analytics.get("uniqueness", 0),
                            cardinality=analytics.get("cardinality", 0),
                            data_type=analytics.get("data_type", "unknown"),
                            missing_percentage=analytics.get("null_percentage", 0)
                            * 100,
                            metadata_text=metadata_text,
                        )
                    except Exception as e:
                        error_msg = f"Error formatting prompt for column '{column_name}': {str(e)}"
                        logger.error(error_msg)
                        state["error_messages"].append(error_msg)

                        # Use rule-based classification if available
                        if column_name in rule_based:
                            llm_classification[column_name] = rule_based[column_name]
                            logger.info(
                                f"Using rule-based classification for '{column_name}' due to prompt error: {rule_based[column_name]['classification']}"
                            )
                            continue
                        else:
                            # Fallback to basic heuristics
                            fallback_classification = (
                                "entity"
                                if uniqueness > UNIQUENESS_THRESHOLD
                                else "property"
                            )
                            llm_classification[column_name] = {
                                "column_name": column_name,
                                "classification": fallback_classification,
                                "confidence": 0.6,
                                "analytics": analytics,
                                "source": "fallback_heuristic",
                            }
                            logger.info(
                                f"Using fallback classification for '{column_name}': {fallback_classification}"
                            )
                            continue

                    # Call the LLM for entity/property classification
                    logger.debug(
                        f"Calling LLM for entity/property classification of column '{column_name}'"
                    )
                    try:
                        # Call the LLM for classification
                        response = call_llm_with_json_output(
                            prompt, state_name="classify_entities_properties"
                        )
                        logger.debug(
                            f"Received LLM classification response for '{column_name}': {response}"
                        )

                        # If response is empty or has an error, use rule-based classification
                        if not response or "error" in response:
                            logger.warning(
                                f"LLM classification failed for '{column_name}', using rule-based classification"
                            )

                            # Use rule-based classification if available
                            if column_name in rule_based:
                                llm_classification[column_name] = rule_based[
                                    column_name
                                ]
                                logger.info(
                                    f"Using rule-based classification for '{column_name}': {rule_based[column_name]['classification']}"
                                )
                                continue
                            else:
                                # Fallback to basic heuristics if rule-based is not available
                                fallback_classification = (
                                    "entity"
                                    if uniqueness > UNIQUENESS_THRESHOLD
                                    else "property"
                                )
                                response = {
                                    "classification": fallback_classification,
                                    "confidence": 0.6,
                                    "reasoning": "Fallback classification based on uniqueness threshold.",
                                }

                        # Extract the classification results
                        classification_result = response.get("classification", "")
                        # Handle the case where classification_result is not a string
                        if not isinstance(classification_result, str):
                            if (
                                isinstance(classification_result, dict)
                                and "type" in classification_result
                            ):
                                classification_result = classification_result["type"]
                            else:
                                # Default to using rule-based classification if available
                                if column_name in rule_based:
                                    classification_result = rule_based[column_name][
                                        "classification"
                                    ]
                                else:
                                    # Fallback based on uniqueness
                                    classification_result = (
                                        "entity"
                                        if uniqueness > UNIQUENESS_THRESHOLD
                                        else "property"
                                    )

                        # Normalize the classification to ensure it's either 'entity' or 'property'
                        classification_result = classification_result.lower()
                        if (
                            "entity" in classification_result
                            and "property" not in classification_result
                        ):
                            classification_result = "entity"
                        elif "property" in classification_result:
                            classification_result = "property"
                        else:
                            # If the classification is ambiguous, use rule-based or analytics to decide
                            if column_name in rule_based:
                                classification_result = rule_based[column_name][
                                    "classification"
                                ]
                                logger.info(
                                    f"Ambiguous LLM classification for '{column_name}', using rule-based: {classification_result}"
                                )
                            else:
                                # Make a decision based on column analytics
                                data_type = analytics.get("data_type", "unknown")

                                # Numeric columns are usually properties
                                if data_type in ["int", "float", "numeric"]:
                                    classification_result = "property"
                                # Date columns are usually properties
                                elif data_type == "datetime":
                                    classification_result = "property"
                                # Email columns are usually properties
                                elif "email" in column_name.lower():
                                    classification_result = "property"
                                # ID columns are usually properties
                                elif "id" in column_name.lower():
                                    classification_result = "property"
                                # High uniqueness suggests entity
                                elif uniqueness > UNIQUENESS_THRESHOLD:
                                    classification_result = "entity"
                                # Default to property for safety
                                else:
                                    classification_result = "property"

                                logger.info(
                                    f"Ambiguous LLM classification for '{column_name}', using analytics-based decision: {classification_result}"
                                )

                        # Double-check that we have a valid classification
                        if classification_result not in ["entity", "property"]:
                            logger.warning(
                                f"Invalid classification '{classification_result}' for '{column_name}', defaulting to 'property'"
                            )
                            classification_result = "property"

                        # Handle confidence value
                        confidence = response.get("confidence", 0.0)
                        if not isinstance(confidence, (int, float)):
                            try:
                                confidence = float(confidence)
                            except (ValueError, TypeError):
                                confidence = 0.5  # Default confidence

                        llm_classification[column_name] = {
                            "column_name": column_name,
                            "classification": classification_result,
                            "confidence": confidence,
                            "analytics": analytics,
                            "source": "llm",
                        }

                        logger.info(
                            f"Classified '{column_name}' as '{classification_result}' with confidence '{confidence}' using LLM approach"
                        )
                    except Exception as e:
                        # If LLM classification fails, use rule-based classification
                        logger.warning(
                            f"LLM classification failed for '{column_name}': {str(e)}"
                        )

                        # Use rule-based classification if available
                        if column_name in rule_based:
                            llm_classification[column_name] = rule_based[column_name]
                            logger.info(
                                f"Using rule-based classification for '{column_name}': {rule_based[column_name]['classification']}"
                            )
                        else:
                            # Use a fallback classification based on column name and analytics
                            entity_keywords = [
                                "id",
                                "code",
                                "city",
                                "country",
                                "state",
                                "region",
                                "category",
                            ]
                            is_likely_entity = any(
                                keyword in column_name.lower()
                                for keyword in entity_keywords
                            )

                            fallback_classification = (
                                "entity"
                                if is_likely_entity or uniqueness > UNIQUENESS_THRESHOLD
                                else "property"
                            )
                            confidence = 0.7 if is_likely_entity else 0.6

                            llm_classification[column_name] = {
                                "column_name": column_name,
                                "classification": fallback_classification,
                                "confidence": confidence,
                                "analytics": analytics,
                                "source": "enhanced_fallback",
                            }
                            logger.info(
                                f"Using enhanced fallback classification for '{column_name}': {fallback_classification}"
                            )
                except Exception as column_error:
                    # If processing a specific column fails, log it and continue with the next one
                    error_msg = (
                        f"Error processing column '{column_name}': {str(column_error)}"
                    )
                    logger.error(error_msg)
                    state["error_messages"].append(error_msg)

                    # Use rule-based classification if available
                    if column_name in rule_based:
                        llm_classification[column_name] = rule_based[column_name]
                    else:
                        # Use a default classification to ensure the pipeline continues
                        llm_classification[column_name] = {
                            "column_name": column_name,
                            "classification": "error",
                            "confidence": 0.5,
                            "analytics": analytics if "analytics" in locals() else {},
                            "source": "error_fallback",
                        }

            # If we processed at least one column successfully, consider it a success
            if processed_columns > 0:
                logger.info(
                    f"Successfully classified {processed_columns} columns, skipped {skipped_columns}"
                )
            else:
                logger.warning("No columns were successfully classified")

        except Exception as e:
            logger.error(f"Error in classification process: {str(e)}")
            # Create a minimal classification for each column to allow the pipeline to continue
            for column_name in state["final_header"]:
                if column_name not in llm_classification:
                    # Use rule-based classification if available, otherwise default to error
                    if column_name in rule_based_classification:
                        llm_classification[column_name] = rule_based_classification[
                            column_name
                        ]
                    else:
                        llm_classification[column_name] = {
                            "column_name": column_name,
                            "classification": "error",
                            "confidence": 0.5,
                            "analytics": state.get("column_analytics", {}).get(
                                column_name, {}
                            ),
                            "source": "global_error_fallback",
                        }

        # Update the state with the LLM classification
        state["entity_property_classification"] = llm_classification

        # We don't need to generate a new rule-based classification here
        # as we're using the one from the column analytics module

    except Exception as e:
        logger.error(f"Error classifying entities/properties: {str(e)}")
        state["error_messages"].append(
            f"Error classifying entities/properties: {str(e)}"
        )

        # Even in case of error, try to use rule-based classification from column analytics
        state["entity_property_classification"] = {}

        try:
            # Use the rule-based classification from column analytics if available
            if (
                "rule_based_classification" in state
                and state["rule_based_classification"]
            ):
                state["entity_property_classification"] = state[
                    "rule_based_classification"
                ]
                logger.info(
                    "Using rule-based classification from column analytics as fallback"
                )
        except Exception as fallback_error:
            logger.error(
                f"Error using rule-based classification as fallback: {str(fallback_error)}"
            )

    return state
