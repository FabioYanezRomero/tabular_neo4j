"""
Property mapping module for entity inference in the Tabular to Neo4j converter.
This module handles mapping properties to their respective entities.
"""

from typing import Dict, Any, List
from Tabular_to_Neo4j.app_state import GraphState
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.nodes.entity_inference.utils import to_neo4j_property_name
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.utils.metadata_utils import (
    load_metadata_for_csv,
    format_metadata_for_prompt,
)
from Tabular_to_Neo4j.utils.csv_utils import get_sample_rows
from Tabular_to_Neo4j.utils.analytics_utils import analyze_column
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS

# Configure logging
logger = get_logger(__name__)


def map_properties_to_entities_node(
    state: GraphState, config: RunnableConfig
) -> GraphState:
    """
    Map properties to their respective entities and create a clear property-entity mapping.

    Args:
        state: The current graph state
        config: LangGraph runnable configuration

    Returns:
        Updated graph state with property_entity_mapping
    """
    logger.info("Starting property-to-entity mapping process")

    # Initialize error messages list if it doesn't exist
    if "error_messages" not in state:
        state["error_messages"] = []

    try:
        # Get the reconciled entity-property classification
        classification = state.get("entity_property_consensus", {})
        if not classification:
            error_msg = (
                "Cannot map properties to entities: missing reconciled classification"
            )
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            # Defensive: Always return a GraphState, never a dict
            if not isinstance(state, GraphState):
                state = GraphState(**dict(state))
            # Ensure the returned state is always a GraphState instance
            if not isinstance(state, GraphState):
                state = GraphState.from_dict(dict(state))
            return state

        # Get the metadata if available
        metadata = None
        if "csv_file_path" in state:
            csv_file_path = state["csv_file_path"]
            metadata = load_metadata_for_csv(csv_file_path)

        # Separate entities and properties
        entities = []
        properties = []

        for column_name, info in classification.items():
            column_class = info.get("classification", "")

            if column_class == "entity":
                entities.append(column_name)
                logger.debug(f"Identified entity column: {column_name}")

            elif column_class == "property":
                properties.append(column_name)
                logger.debug(f"Identified property column: {column_name}")

        logger.info(f"Found {len(entities)} entities and {len(properties)} properties")

        # If no entities or properties, return the state unchanged
        if not entities:
            logger.warning("No entities found, skipping property mapping")
            # Defensive: Always return a GraphState, never a dict
            if not isinstance(state, GraphState):
                state = GraphState(**dict(state))
            # Ensure the returned state is always a GraphState instance
            if not isinstance(state, GraphState):
                state = GraphState.from_dict(dict(state))
            return state

        # Use the first entity as our main entity for properties
        main_entity = entities[0]
        logger.info(f"Using {main_entity} as the main entity for properties")

        # Create a property-entity mapping that includes ALL entities
        property_entity_mapping = {}

        # Add all entities to the mapping
        for entity in entities:
            property_entity_mapping[entity] = {
                "type": "entity",
                "properties": [],
                "is_primary": entity == main_entity,  # Mark the first entity as primary
            }
            logger.info(f"Added entity '{entity}' to mapping")

        # Add all properties to the main entity
        for prop in properties:
            property_key = to_neo4j_property_name(prop)
            property_entity_mapping[main_entity]["properties"].append(
                {"column_name": prop, "property_key": property_key}
            )
            logger.info(f"Mapped property '{prop}' to entity '{main_entity}'")

        # Format a sample prompt for documentation purposes
        # Get the processed dataframe for sample data if available
        sample_data = ""
        if state.get("processed_dataframe") is not None:
            df = state["processed_dataframe"]
            if not df.empty:
                # Get a small sample of the data (first MAX_SAMPLE_ROWS rows)
                try:
                    sample_data = df.head(MAX_SAMPLE_ROWS).to_dict(orient="records")
                    sample_data = str(sample_data)  # Convert to string for the prompt
                except Exception as e:
                    logger.warning(f"Error converting dataframe to dict: {str(e)}")
                    # Fallback to string representation
                    sample_data = df.head(MAX_SAMPLE_ROWS).to_string(index=False)

        # Get metadata text if available
        metadata_text = (
            format_metadata_for_prompt(metadata)
            if metadata
            else "No metadata available."
        )

        # Format the prompt with entity and property information
        prompt = format_prompt(
            "map_properties_to_entities.txt",
            entity_property_classification=str(classification),
            entities=str(entities),
            properties=str(properties),
            metadata_text=metadata_text,
            sample_data=sample_data,
        )

        # Save the prompt sample without actually calling the LLM
        from Tabular_to_Neo4j.utils.prompt_utils import save_prompt_sample

        from Tabular_to_Neo4j.utils.output_saver import get_output_saver
        output_saver = get_output_saver()
        base_dir = output_saver.base_dir if output_saver else "samples"
        timestamp = output_saver.timestamp if output_saver else None
        save_prompt_sample(
            "map_properties_to_entities.txt",
            prompt,
            {"state_name": "map_properties_to_entities"},
            base_dir=base_dir,
            timestamp=timestamp,
            subfolder="prompts",
        )

        # Save the result as if it was an LLM response
        result_json = {"property_entity_mapping": property_entity_mapping}
        save_prompt_sample(
            "map_properties_to_entities_result.txt",
            str(result_json),
            {"state_name": "map_properties_to_entities"},
            base_dir=base_dir,
            timestamp=timestamp,
            subfolder="prompts",
        )

        # Update the state with the final mapping
        state["property_entity_mapping"] = property_entity_mapping

    except Exception as e:
        logger.error(f"Error in property mapping: {str(e)}")
        state["error_messages"].append(f"Error in property mapping: {str(e)}")
        state["property_entity_mapping"] = {}

    # Ensure the returned state is always a GraphState instance
    if not isinstance(state, GraphState):
        state = GraphState.from_dict(dict(state))
    return state
