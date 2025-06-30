"""
Relationship inference module for entity inference in the Tabular to Neo4j converter.
This module handles inferring relationships between entity types.
"""

from typing import Dict, Any, List
from Tabular_to_Neo4j.app_state import GraphState
import os
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import (
    get_metadata_for_state,
    format_metadata_for_prompt,
)
from Tabular_to_Neo4j.utils.logging_config import get_logger
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS
from Tabular_to_Neo4j.utils.llm_manager import get_node_order_for_state

# Configure logging
logger = get_logger(__name__)


def infer_entity_relationships_node(
    state: GraphState, node_order: int
) -> GraphState:
    """
    Infer relationships between entities based on property mapping and entity types.
    Uses LLM to determine the most likely relationships between entities.

    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline

    Returns:
        Updated graph state with entity_relationships
    """
    logger.info("Starting entity relationship inference process")

    # Validate required inputs
    if state.get("property_entity_mapping") is None:
        error_msg = "Cannot infer entity relationships: missing property-entity mapping"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        # Ensure the returned state is always a GraphState instance
        if not isinstance(state, GraphState):
            state = GraphState.from_dict(dict(state))
        return state

    try:
        # Get the property-entity mapping
        mapping = state["property_entity_mapping"]

        # Get column analytics if available
        column_analytics = state.get("column_analytics", {})

        # Extract entity types with their properties and analytics
        entities_with_properties = []
        for entity_type, entity_data in mapping.items():
            if entity_data.get("type") == "entity":
                properties = entity_data.get("properties", [])

                # Enhance properties with analytics data if available
                enhanced_properties = []
                for prop in properties:
                    column_name = prop.get("column_name", "")
                    enhanced_prop = prop.copy()

                    # Add analytics data if available
                    if column_name in column_analytics:
                        analytics = column_analytics[column_name]
                        enhanced_prop["analytics"] = {
                            "data_type": analytics.get("data_type", "unknown"),
                            "uniqueness_ratio": analytics.get("uniqueness_ratio", 0),
                            "cardinality": analytics.get("cardinality", 0),
                            "missing_percentage": analytics.get(
                                "missing_percentage", 0
                            ),
                            "sample_values": analytics.get("sample_values", []),
                        }

                    enhanced_properties.append(enhanced_prop)

                entities_with_properties.append(
                    {"entity_type": entity_type, "properties": enhanced_properties}
                )

        entity_types = [entity["entity_type"] for entity in entities_with_properties]
        logger.info(
            f"Found {len(entity_types)} entity types for relationship inference: {', '.join(entity_types)}"
        )

        # If we have 0 or 1 entity types, there are no relationships to infer
        if len(entity_types) <= 1:
            logger.info(
                f"Found {len(entity_types)} entity types, no relationships to infer"
            )

            # Not enough entity types to call LLM; do not save prompt, just update state
            state["entity_relationships"] = []
            if not isinstance(state, GraphState):
                state = GraphState.from_dict(dict(state))
            return state

        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = (
            format_metadata_for_prompt(metadata)
            if metadata
            else "No metadata available."
        )

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

        # Only format and save the prompt if there are at least 2 entity types
        if len(entity_types) > 1:
            
            # Extract table_name from csv_file_path if possible
            table_name = os.path.splitext(os.path.basename(state.get("csv_file_path", "")))[0]

            # Use a pairwise approach to infer relationships between each pair of entities
            all_relationships = []
            processed_pairs = set()  # Track which pairs we've already processed

            # For each pair of entities, infer the relationship between them
            for i, entity1 in enumerate(entity_types):
                for j, entity2 in enumerate(entity_types):
                    # Skip self-relationships
                    if i == j:
                        continue

                    # Create a pair identifier (sorted to ensure we don't process the same pair twice)
                    pair = tuple(sorted([entity1, entity2]))

                    # Skip if we've already processed this pair
                    if pair in processed_pairs:
                        continue

                    processed_pairs.add(pair)
                    logger.info(
                        f"Inferring relationship between entities: {pair[0]} and {pair[1]}"
                    )

                    # Format a focused prompt specifically for this entity pair
                    prompt = format_prompt(
                        "infer_entity_relationship_pair.txt",
                        table_name=table_name,
                        source_entity=pair[0],
                        target_entity=pair[1],
                        entity_property_consensus=str(
                            state.get("entity_property_consensus", {})
                        ),
                        property_entity_mapping=str(mapping),
                        metadata_text=metadata_text,
                        sample_data=sample_data,
                        unique_suffix=f"{pair[0]}_{pair[1]}",
                    )
                    
                    # Call the LLM for this specific entity pair
                    state_name = f"infer_relationship_{pair[0]}_{pair[1]}"
                    response = call_llm_with_json_output(
                        prompt=prompt,
                        state_name=state_name,
                        unique_suffix=f"{pair[0]}_{pair[1]}",
                        table_name=table_name,
                        template_name="infer_entity_relationship_pair.txt",
                        node_order=node_order
                    )
                    
                    # Extract the inferred relationship
                    response_relationships = response.get("entity_relationships", [])
                    valid_relationships = []
                    for rel in response_relationships:
                        source = rel.get("source_entity", "")
                        target = rel.get("target_entity", "")
                        rel_type = rel.get("relationship_type", "")
                        confidence = rel.get("confidence", 0.0)
                        bidirectional = rel.get("bidirectional", False)
                        reasoning = rel.get("reasoning", "")

                        # Validate the relationship has required fields and sufficient confidence
                        if source and target and rel_type and confidence > 0.5:
                            valid_relationships.append(rel)
                            logger.info(
                                f"Valid relationship: ({source})-[{rel_type}]->({target}) with confidence {confidence}"
                            )
                            if reasoning:
                                logger.info(f"Reasoning: {reasoning}")
                            if bidirectional:
                                reverse_rel = {
                                    "source_entity": target,
                                    "target_entity": source,
                                    "relationship_type": rel_type,
                                    "confidence": confidence,
                                    "bidirectional": True,
                                    "reasoning": f"Reverse of bidirectional relationship: {reasoning}",
                                }
                                valid_relationships.append(reverse_rel)
                                logger.info(
                                    f"Added reverse relationship: ({target})-[{rel_type}]->({source}) (bidirectional)"
                                )
                        else:
                            logger.warning(
                                f"Filtered out invalid relationship: {rel} - missing required fields or low confidence"
                            )
                    all_relationships.extend(valid_relationships)

            # Update the state with all valid relationships
            state["entity_relationships"] = all_relationships
        else:
            # Not enough entity types to call LLM; do not save prompt, just update state
            logger.info(
                f"Found {len(entity_types)} entity types, no relationships to infer"
            )
            state["entity_relationships"] = []
            if not isinstance(state, GraphState):
                state = GraphState.from_dict(dict(state))
            return state

    except Exception as e:
        logger.error(f"Error inferring entity relationships: {str(e)}")
        state["error_messages"].append(
            f"Error inferring entity relationships: {str(e)}"
        )
        state["entity_relationships"] = []

    # Ensure the returned state is always a GraphState instance
    if not isinstance(state, GraphState):
        state = GraphState.from_dict(dict(state))
    return state