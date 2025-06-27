"""
Cypher generation module for database schema generation in the Tabular to Neo4j converter.
This module handles generating Cypher query templates.
"""

from typing import Dict, Any, List
import os
import json
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.llm_manager import call_llm_with_json_output
from Tabular_to_Neo4j.utils.metadata_utils import (
    get_metadata_for_state,
    format_metadata_for_prompt,
)
from Tabular_to_Neo4j.config import MAX_SAMPLE_ROWS
from Tabular_to_Neo4j.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def generate_cypher_templates_node(
    state: GraphState, config: RunnableConfig
) -> GraphState:
    """
    Generate template Cypher queries for the inferred schema.
    Uses LLM to create Cypher query templates for loading and querying the Neo4j graph model.

    Args:
        state: The current graph state
        config: LangGraph runnable configuration

    Returns:
        Updated graph state with cypher_query_templates
    """
    logger.info("Starting Cypher template generation process")

    # Validate required inputs
    missing_inputs = []
    if state.get("property_entity_mapping") is None:
        missing_inputs.append("property_entity_mapping")
    if state.get("entity_relationships") is None:
        missing_inputs.append("entity_relationships")

    if missing_inputs:
        error_msg = f"Cannot generate Cypher templates: missing required data: {', '.join(missing_inputs)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        return state

    try:
        # Get the property-entity mapping and relationships
        mapping = state["property_entity_mapping"]
        relationships = state["entity_relationships"]

        # Extract entity types and their properties
        entities = {}
        for entity_name, item in mapping.items():
            if item.get("type") == "entity":
                # Use the column name as the entity type if entity_type is not specified
                entity_type = item.get("entity_type", entity_name)

                # Get properties for this entity
                properties = item.get("properties", [])

                # Every entity needs a generated UUID as a unique identifier
                entities[entity_type] = {
                    "is_primary": item.get(
                        "is_primary", True
                    ),  # Default to primary if not specified
                    "properties": properties,
                    "needs_generated_id": True,  # Always generate a unique ID for every entity
                }

        logger.debug(
            f"Preparing Cypher templates for {len(entities)} entities and {len(relationships)} relationships"
        )

        # Get metadata for the CSV file
        metadata = get_metadata_for_state(state)
        metadata_text = (
            format_metadata_for_prompt(metadata)
            if metadata
            else "No metadata available."
        )

        # Get sample data from the processed dataframe
        sample_data = ""
        if state.get("processed_dataframe") is not None:
            df = state["processed_dataframe"]
            if not df.empty:
                try:
                    sample_data = df.head(MAX_SAMPLE_ROWS).to_dict(orient="records")
                    sample_data = str(sample_data)  # Convert to string for the prompt
                except Exception as e:
                    logger.warning(f"Error converting dataframe to dict: {str(e)}")
                    # Fallback to string representation
                    sample_data = df.head(MAX_SAMPLE_ROWS).to_string(index=False)

        # Use LLM to generate Cypher templates
        logger.info("Using LLM to generate Cypher templates")

        try:
            # Format the prompt for the LLM
            template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "prompts",
                "generate_cypher_templates.txt",
            )

            # Format entities for the prompt
            entities_for_prompt = {}
            for entity_type, entity_info in entities.items():
                properties_list = []
                for prop in entity_info.get("properties", []):
                    if (
                        isinstance(prop, dict)
                        and "column_name" in prop
                        and "property_key" in prop
                    ):
                        properties_list.append(
                            {
                                "column_name": prop["column_name"],
                                "property_key": prop["property_key"],
                            }
                        )

                entities_for_prompt[entity_type] = {
                    "is_primary": entity_info.get("is_primary", True),
                    "properties": properties_list,
                }

            # Format relationships for the prompt
            relationships_for_prompt = []
            for rel in relationships:
                relationships_for_prompt.append(
                    {
                        "source_entity": rel.get("source_entity"),
                        "target_entity": rel.get("target_entity"),
                        "relationship_type": rel.get("relationship_type"),
                        "confidence": rel.get("confidence", 0.0),
                        "reasoning": rel.get("reasoning", ""),
                    }
                )

            # Format the prompt with the template
            table_name = state.get("table_name")
            prompt = format_prompt(
                template_name="generate_cypher_templates.txt",
                table_name=table_name,
                entities=json.dumps(entities_for_prompt, indent=2),
                relationships=json.dumps(relationships_for_prompt, indent=2),
                metadata_text=metadata_text,
                sample_data=sample_data,
            )

            # Call the LLM to generate Cypher templates
            response = call_llm_with_json_output(
                prompt, state_name="generate_cypher_templates"
            )

            # Initialize the query lists if they don't exist in the response
            if not response:
                response = {}
            if "entity_creation_queries" not in response:
                response["entity_creation_queries"] = []
            if "relationship_queries" not in response:
                response["relationship_queries"] = []
            if "example_queries" not in response:
                response["example_queries"] = []
            if "constraint_queries" not in response:
                response["constraint_queries"] = []

            # Generate rule-based Cypher templates as a fallback
            def generate_rule_based_templates():
                # Initialize response structure
                rule_based_response = {
                    "entity_creation_queries": [],
                    "relationship_queries": [],
                    "example_queries": [],
                    "constraint_queries": [],
                }

                # Process each entity to generate Cypher queries
                for entity_type, entity_info in entities.items():
                    properties = entity_info.get("properties", [])

                    # Generate constraint query for entity
                    constraint_query = (
                        f"CREATE CONSTRAINT ON (n:{entity_type}) ASSERT n.id IS UNIQUE;"
                    )
                    rule_based_response["constraint_queries"].append(
                        {
                            "query": constraint_query,
                            "description": f"Create unique constraint on {entity_type}.id",
                        }
                    )

                    # Generate entity creation query
                    property_assignments = []
                    for prop in properties:
                        if (
                            isinstance(prop, dict)
                            and "column_name" in prop
                            and "property_key" in prop
                        ):
                            column_name = prop["column_name"]
                            property_key = prop["property_key"]
                            property_assignments.append(
                                f"{property_key}: row.{column_name}"
                            )

                    # Always add a unique ID
                    property_assignments.append("id: randomUUID()")

                    property_assignments_str = ", ".join(property_assignments)
                    entity_query = f"LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row\nCREATE (n:{entity_type} {{{property_assignments_str}}});"

                    rule_based_response["entity_creation_queries"].append(
                        {
                            "query": entity_query,
                            "description": f"Create {entity_type} nodes from CSV data",
                        }
                    )

                    # Generate example query
                    example_query = f"MATCH (n:{entity_type}) RETURN n LIMIT 5;"
                    rule_based_response["example_queries"].append(
                        {
                            "query": example_query,
                            "description": f"Retrieve sample {entity_type} nodes",
                        }
                    )

                # Process relationships
                for rel in relationships:
                    source_entity = rel.get("source_entity")
                    target_entity = rel.get("target_entity")
                    relationship_type = rel.get("relationship_type")

                    if source_entity and target_entity and relationship_type:
                        rel_query = f"MATCH (source:{source_entity}), (target:{target_entity})\nWHERE source.id = row.source_id AND target.id = row.target_id\nCREATE (source)-[:{relationship_type}]->(target);"

                        rule_based_response["relationship_queries"].append(
                            {
                                "query": rel_query,
                                "description": f"Create {relationship_type} relationships between {source_entity} and {target_entity}",
                            }
                        )

                return rule_based_response

            # If the response is empty or invalid, fall back to rule-based generation
            if not response or not isinstance(response, dict):
                logger.warning(
                    "LLM returned invalid response for Cypher templates, falling back to rule-based generation"
                )
                response = generate_rule_based_templates()
            else:
                # Ensure all required keys exist in the response
                if "entity_creation_queries" not in response:
                    response["entity_creation_queries"] = []
                if "relationship_queries" not in response:
                    response["relationship_queries"] = []
                if "example_queries" not in response:
                    response["example_queries"] = []
                if "constraint_queries" not in response:
                    response["constraint_queries"] = []

                # If any of the query lists are empty, generate them using rule-based approach
                if (
                    not response["entity_creation_queries"]
                    or not response["relationship_queries"]
                ):
                    logger.warning(
                        "LLM returned incomplete Cypher templates, filling in missing parts with rule-based generation"
                    )
                    rule_based = generate_rule_based_templates()

                    # Fill in any missing query types
                    if not response["entity_creation_queries"]:
                        response["entity_creation_queries"] = rule_based[
                            "entity_creation_queries"
                        ]
                    if not response["relationship_queries"]:
                        response["relationship_queries"] = rule_based[
                            "relationship_queries"
                        ]
                    if not response["example_queries"]:
                        response["example_queries"] = rule_based["example_queries"]
                    if not response["constraint_queries"]:
                        response["constraint_queries"] = rule_based[
                            "constraint_queries"
                        ]

            # Update the state with the generated Cypher templates
            state["cypher_query_templates"] = response

            # Log the number of queries generated
            logger.info(
                f"Generated {len(response.get('entity_creation_queries', []))} entity creation queries, {len(response.get('relationship_queries', []))} relationship queries, and {len(response.get('example_queries', []))} example queries"
            )

            return state

        except Exception as e:
            error_msg = f"Error in Cypher template generation process: {str(e)}"
            logger.error(error_msg)
            state["error_messages"].append(error_msg)
            return state

    except Exception as e:
        error_msg = f"Error in Cypher template generation process: {str(e)}"
        logger.error(error_msg)
        state["error_messages"].append(error_msg)
        return state
