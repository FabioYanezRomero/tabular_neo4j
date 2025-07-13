"""
Entity reconciliation module for entity inference in the Tabular to Neo4j converter.
This module handles reconciling different classification approaches.
"""

from typing import Dict, Any
from Tabular_to_Neo4j.app_state import GraphState
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.config import UNIQUENESS_THRESHOLD
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def reconcile_entity_property_node(
    state: GraphState, node_order: int, use_analytics: bool = False
) -> GraphState:
    """
    Reconcile analytics-based and LLM-based classifications
    to create a consensus model of entities and properties.
    Uses LLM to reconcile different classification approaches.

    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline

    Returns:
        Updated graph state with entity_property_consensus
    """
    logger.info("Starting entity-property reconciliation process")

    # Validate required inputs
    if state.get("entity_property_classification") is None:
        error_msg = "Cannot reconcile entity/property classifications: missing entity_property_classification"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        # Defensive: Always return a GraphState, never a dict
        if not isinstance(state, GraphState):
            state = GraphState(**dict(state))
        # Ensure the returned state is always a GraphState instance
        if not isinstance(state, GraphState):
            state = GraphState.from_dict(dict(state))
        print("[DEBUG] reconcile_entity_property_node: llm_classification keys:", list(locals().get('llm_classification', {}).keys()) if 'llm_classification' in locals() else 'N/A')
        print("[DEBUG] reconcile_entity_property_node: rule_based_classification keys:", list(locals().get('rule_based_classification', {}).keys()) if 'rule_based_classification' in locals() else 'N/A')
        print("[DEBUG] reconcile_entity_property_node: consensus keys:", list(state.get("entity_property_consensus", {}).keys()))
        return state

    # Debug: Log state keys to understand what's available
    logger.info(f"State keys: {list(state.keys())}")

    # Check if rule-based classification is available from column analytics
    has_rule_based = (
        "column_analytics" in state
        and "rule_based_classification" in state.get("column_analytics", {})
    )
    if has_rule_based:
        logger.info(f"Rule-based classification exists in column analytics")
    else:
        logger.info(
            "Rule-based classification is missing in column analytics - will use only LLM classification"
        )

    try:
        # Get the LLM classification
        llm_classification = state["entity_property_classification"]

        # Get rule-based classification from column analytics if available
        if has_rule_based:
            rule_based_classification = state["column_analytics"][
                "rule_based_classification"
            ]
            logger.info(
                f"Using rule-based classification from column analytics with {len(rule_based_classification)} entries"
            )
        # Create rule-based classification directly from analytics if not in column analytics
        else:
            logger.info("Creating rule-based classification from analytics")
            rule_based_classification = {}

            for column_name, analytics_data in state.get(
                "column_analytics", {}
            ).items():
                if isinstance(analytics_data, dict) and "column_name" in analytics_data:
                    uniqueness = analytics_data.get("uniqueness", 0)
                    cardinality = analytics_data.get("cardinality", 0)
                    data_type = analytics_data.get("data_type", "")
                    patterns = analytics_data.get("patterns", {})

                    # Force classification as property for specific data types
                    if data_type in ["date", "datetime", "float", "integer"]:
                        classification = "property"
                    # Check for email pattern
                    elif "email" in patterns and patterns["email"] > 0.5:
                        classification = "property"
                    # Check for numeric patterns
                    elif (
                        any(
                            p in patterns
                            for p in ["phone", "credit_card", "numeric_id"]
                        )
                        and patterns.get(
                            next(
                                (
                                    p
                                    for p in patterns
                                    if p in ["phone", "credit_card", "numeric_id"]
                                ),
                                None,
                            ),
                            0,
                        )
                        > 0.5
                    ):
                        classification = "property"
                    # Apply standard rules
                    elif uniqueness > UNIQUENESS_THRESHOLD:
                        classification = "entity"
                    elif (
                        cardinality < len(state.get("processed_dataframe", [])) * 0.1
                        and cardinality > 1
                    ):
                        classification = "entity"
                    else:
                        classification = "property"

                    rule_based_classification[column_name] = {
                        "column_name": column_name,
                        "classification": classification,
                        "analytics": analytics_data,
                        "source": "rule_based",
                    }

            logger.info(
                f"Created rule-based classification for {len(rule_based_classification)} columns"
            )

        # Initialize consensus dictionary
        consensus = {}

        # Process each column to determine if reconciliation is needed
        for column_name in state.get("final_header", []):
            # Get both classifications for this column
            llm_info = llm_classification.get(column_name, {})
            rule_info = rule_based_classification.get(column_name, {})

            # Check if there's a discrepancy between the classifications
            llm_classification_result = llm_info.get("classification") or llm_info.get("consensus_classification", "")
            rule_classification_result = rule_info.get("classification") or rule_info.get("consensus_classification", "")

            # If classifications match, no need for reconciliation
            if llm_classification_result == rule_classification_result:
                logger.info(
                    f"Classifications match for column '{column_name}': {llm_classification_result}"
                )

                # Use the LLM classification as it typically has more detailed reasoning
                consensus[column_name] = llm_info
                continue

            # If there's a discrepancy, perform reconciliation
            logger.info(
                f"Classification discrepancy for '{column_name}': LLM={llm_classification_result}, Rule={rule_classification_result}"
            )

            # --- PROMPT FORMATTING AND SAVING FOR RECONCILIATION ---
            from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
            from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
            from Tabular_to_Neo4j.utils.metadata_utils import format_metadata_for_prompt

            # Gather metadata for prompt
            analytics = state.get("column_analytics", {}).get(column_name, {})
            metadata = analytics
            metadata_text = format_metadata_for_prompt(metadata) if metadata else "No metadata available."
            sample_values = ""
            if state.get("processed_dataframe") is not None:
                df = state["processed_dataframe"]
                if column_name in df.columns:
                    sample_vals = df[column_name].head(5).tolist()
                    sample_values = ", ".join(map(str, sample_vals))
            import os
            table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "")))[0]

            # Format and save the prompt
            prompt = format_prompt(
                template_name="reconcile_entity_property.txt",
                table_name=table_name,
                column_name=column_name,
                rule_based_classification=rule_classification_result,
                llm_classification=llm_classification_result,
                analytics_classification=analytics,
                metadata_text=metadata_text,
                sample_values=sample_values,
                unique_suffix=column_name,
                use_analytics=use_analytics
            )
            logger.info(f"Formatted and saved reconciliation prompt for column '{column_name}' due to classification discrepancy.")

            # Call the LLM with the prompt and update llm_info
            try:
                llm_result = call_llm_with_json_output(
                    prompt=prompt,
                    state_name="reconcile_entity_property",
                    unique_suffix=column_name,
                    table_name=table_name,
                    template_name="reconcile_entity_property.txt",
                    node_order=node_order
                )
                logger.info(f"LLM reconciliation result for '{column_name}': {llm_result}")
                llm_result["analytics"] = analytics
                norm_info = llm_result if isinstance(llm_result, dict) else {"classification": str(llm_result)}
                if "consensus_classification" in norm_info:
                    norm_info["classification"] = norm_info.pop("consensus_classification")
                consensus[column_name] = norm_info
            except Exception as llm_error:
                logger.error(f"Error calling LLM for reconciliation on column '{column_name}': {str(llm_error)}")
               

        # Add uniqueness information for entities to help with later processing
        for column_name, info in consensus.items():
            if info.get("classification", "") == "entity":
                analytics = state.get("column_analytics", {}).get(column_name, {})
                info["uniqueness_ratio"] = analytics.get(
                    "uniqueness", 0
                )  # Updated field name

        # Update the state
        state["entity_property_consensus"] = consensus

    except Exception as e:
        logger.error(f"Error reconciling entity/property classifications: {str(e)}")
        state["error_messages"].append(
            f"Error reconciling entity/property classifications: {str(e)}"
        )
        state["entity_property_consensus"] = state.get(
            "entity_property_classification", {}
        )

    if not isinstance(state, GraphState):
        state = GraphState(**dict(state))
    print("[DEBUG] reconcile_entity_property_node: llm_classification keys:", list(locals().get('llm_classification', {}).keys()) if 'llm_classification' in locals() else 'N/A')
    print("[DEBUG] reconcile_entity_property_node: rule_based_classification keys:", list(locals().get('rule_based_classification', {}).keys()) if 'rule_based_classification' in locals() else 'N/A')
    print("[DEBUG] reconcile_entity_property_node: consensus keys:", list(state.get("entity_property_consensus", {}).keys()))
    return state
