"""
Main script for the Tabular to Neo4j converter using LangGraph.
"""

import argparse
import json
import os
import time
import logging
from typing import Dict, Any
from pathlib import Path
from Tabular_to_Neo4j.utils.logging_config import get_logger, setup_logging
import sys

# Initialize logging with default configuration
setup_logging()

# Get a logger for this module
logger = get_logger(__name__)

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")


from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from Tabular_to_Neo4j.app_state import GraphState

# Import input nodes
from Tabular_to_Neo4j.pipeline_config import (
    PIPELINE_NODES,
    PIPELINE_EDGES,
    ENTRY_POINT,
)

# Import output saver
from Tabular_to_Neo4j.utils.output_saver import (
    initialize_output_saver,
    get_output_saver,
)

# Get a logger for this module
logger = get_logger(__name__)


def create_graph() -> StateGraph:
    """
    Create the LangGraph for CSV analysis and Neo4j schema inference.

    Returns:
        StateGraph instance
    """
    # Create the graph with the GraphState type
    graph = StateGraph(GraphState)

    # Get the output saver
    output_saver = get_output_saver()

    # Determine node order based on configuration
    node_order = {name: idx + 1 for idx, (name, _) in enumerate(PIPELINE_NODES)}

    # Add nodes using the declarative configuration
    for node_name, node_func in PIPELINE_NODES:
        graph.add_node(
            node_name,
            wrap_node_with_output_saving(node_name, node_func, node_order[node_name]),
        )

    # Define edges using the declarative configuration
    for edge in PIPELINE_EDGES:
        if isinstance(edge, tuple):
            graph.add_edge(edge[0], edge[1])
        else:
            graph.add_conditional_edges(edge["from"], edge["condition"], edge["edges"])

    # Set the entry point
    graph.set_entry_point(ENTRY_POINT)

    return graph


def format_schema_output(schema: Dict[str, Any]) -> str:
    """
    Format the inferred Neo4j schema for human-readable output.

    Args:
        schema: The inferred Neo4j schema

    Returns:
        Formatted string representation
    """
    output = []

    # Primary entity
    output.append(f"PRIMARY ENTITY: :{schema['primary_entity_label']}")
    output.append("")

    # Group columns by role
    columns_by_role = {}
    for col in schema.get("columns_classification", []):
        role = col.get("role", "UNKNOWN")
        if role not in columns_by_role:
            columns_by_role[role] = []
        columns_by_role[role].append(col)

    # Format primary entity identifiers
    if "PRIMARY_ENTITY_IDENTIFIER" in columns_by_role:
        output.append("PRIMARY ENTITY IDENTIFIERS:")
        for col in columns_by_role["PRIMARY_ENTITY_IDENTIFIER"]:
            output.append(
                f"  - {col['original_column_name']} → .{col['neo4j_property_key']}"
            )
        output.append("")

    # Format primary entity properties
    if "PRIMARY_ENTITY_PROPERTY" in columns_by_role:
        output.append("PRIMARY ENTITY PROPERTIES:")
        for col in columns_by_role["PRIMARY_ENTITY_PROPERTY"]:
            note = f" (Note: {col['note']})" if "note" in col else ""
            output.append(
                f"  - {col['original_column_name']} → .{col['neo4j_property_key']}{note}"
            )
        output.append("")

    # Format new node types
    if "NEW_NODE_TYPE_VALUES" in columns_by_role:
        output.append("NEW NODE TYPES:")
        for col in columns_by_role["NEW_NODE_TYPE_VALUES"]:
            output.append(
                f"  - {col['original_column_name']} → :{col['new_node_label']} nodes"
            )
            output.append(
                f"    Relationship: (:{schema['primary_entity_label']})-[:{col['relationship_to_primary']}]->(:{col['new_node_label']})"
            )

            # Add associated properties for this node type
            if "NEW_NODE_PROPERTY" in columns_by_role:
                associated_props = [
                    prop
                    for prop in columns_by_role["NEW_NODE_PROPERTY"]
                    if prop.get("associated_new_node_label") == col["new_node_label"]
                ]
                if associated_props:
                    output.append(f"    Properties:")
                    for prop in associated_props:
                        output.append(
                            f"      - {prop['original_column_name']} → .{prop['neo4j_property_key']}"
                        )

            output.append("")

    # Format relationship properties
    if "RELATIONSHIP_PROPERTY" in columns_by_role:
        output.append("RELATIONSHIP PROPERTIES:")
        for col in columns_by_role["RELATIONSHIP_PROPERTY"]:
            rel = col.get("associated_relationship", "UNKNOWN")
            source = col.get("source_node_label", schema["primary_entity_label"])
            target = col.get("target_node_label", "UNKNOWN")
            output.append(
                f"  - {col['original_column_name']} → property on relationship (:{source})-[:{rel}]->(:{target})"
            )
        output.append("")

    return "\n".join(output)


def run_analysis(
    csv_file_path: str,
    output_file: str = None,
    verbose: bool = False,
    save_node_outputs: bool = False,
    output_dir: str = "samples",
) -> Dict[str, Any]:
    """
    Run the CSV analysis and Neo4j schema inference.

    Args:
        csv_file_path: Path to the CSV file
        output_file: Optional path to save the results
        verbose: Whether to print verbose output

    Returns:
        The final graph state
    """
    # Import and reset the prompt sample directory to ensure all samples from this run
    # are stored in the same directory
    from Tabular_to_Neo4j.utils.prompt_utils import reset_prompt_sample_directory
    from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver, get_output_saver

    # Always initialize OutputSaver to guarantee a single timestamp per run
    initialize_output_saver(output_dir)
    output_saver = get_output_saver()
    # Use the OutputSaver timestamp for all prompt samples and outputs
    reset_prompt_sample_directory(base_dir=output_dir, timestamp=output_saver.timestamp if output_saver else None)

    # Log analysis start with file details
    logger.info(f"Starting analysis of CSV file: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    # Log file size and basic info
    file_size = os.path.getsize(csv_file_path) / 1024  # KB
    logger.info(f"File size: {file_size:.2f} KB")

    # Initialize output saver if requested
    if save_node_outputs:
        logger.info(f"Initializing output saver with directory: {output_dir}")
        initialize_output_saver(output_dir)

    # Create the graph
    logger.debug("Creating state graph")
    graph = create_graph()

    # Compile the graph
    logger.debug("Compiling state graph")
    app = graph.compile()

    # Set up the initial state
    logger.debug("Initializing state")
    initial_state = GraphState(csv_file_path=csv_file_path, error_messages=[])

    # Run the graph
    logger.info(f"Executing analysis pipeline")
    start_time = time.time()
    try:
        final_state = app.invoke(initial_state)
        execution_time = time.time() - start_time
        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

    # Log the results
    if final_state.get("error_messages"):
        error_count = len(final_state.get("error_messages", []))
        logger.warning(f"Analysis completed with {error_count} errors/warnings")
        for i, error in enumerate(final_state.get("error_messages", [])):
            logger.warning(f"Error/Warning {i+1}: {error}")
    else:
        logger.info("Analysis completed successfully with no errors")

    # Log cypher template information
    if final_state.get("cypher_query_templates"):
        templates = final_state["cypher_query_templates"]
        entity_count = len(templates.get("entity_creation_queries", []))
        relationship_count = len(templates.get("relationship_queries", []))
        example_count = len(templates.get("example_queries", []))
        logger.info(
            f"Generated {entity_count} entity creation queries, {relationship_count} relationship queries, and {example_count} example queries"
        )
    else:
        logger.warning("No Cypher templates were generated")

    # Save results to file if requested
    if output_file:
        try:
            logger.info(f"Saving results to {output_file}")
            with open(output_file, "w") as f:
                if final_state.get("cypher_query_templates"):
                    templates = final_state["cypher_query_templates"]
                    f.write("ENTITY CREATION QUERIES:\n")
                    for i, query in enumerate(
                        templates.get("entity_creation_queries", [])
                    ):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")

                    f.write("\nRELATIONSHIP QUERIES:\n")
                    for i, query in enumerate(
                        templates.get("relationship_queries", [])
                    ):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")

                    f.write("\nEXAMPLE QUERIES:\n")
                    for i, query in enumerate(templates.get("example_queries", [])):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                else:
                    f.write("No Cypher templates generated.\n")

                if final_state.get("error_messages"):
                    f.write("\nErrors/Warnings:\n")
                    for error in final_state.get("error_messages", []):
                        f.write(f"  - {error}\n")
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {str(e)}")

    # Print the results if verbose
    if verbose:
        logger.info("Displaying analysis results")

        # Log any errors
        if final_state.get("error_messages"):
            logger.info("\nErrors/Warnings encountered during analysis:")
            for error in final_state.get("error_messages", []):
                logger.info("  - %s", error)

        # Save the final state as individual JSON files in state/<timestamp>/
        try:
            from Tabular_to_Neo4j.utils.state_saver import save_state_snapshot
            from Tabular_to_Neo4j.utils.output_saver import get_output_saver
            output_saver = get_output_saver()
            base_dir = output_saver.base_dir if output_saver else "samples"
            timestamp = output_saver.timestamp if output_saver else None
            save_state_snapshot(final_state, timestamp=timestamp, base_dir=base_dir)
        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")

        # Log the generated Cypher templates
        if final_state.get("cypher_query_templates"):
            templates = final_state["cypher_query_templates"]
            logger.info("\nGenerated Cypher Templates:")

            logger.info("\nENTITY CREATION QUERIES:")
            for i, query in enumerate(templates.get("entity_creation_queries", [])):
                logger.info("\nQuery %s:\n%s", i + 1, query.get("query", ""))

            logger.info("\nRELATIONSHIP QUERIES:")
            for i, query in enumerate(templates.get("relationship_queries", [])):
                logger.info("\nQuery %s:\n%s", i + 1, query.get("query", ""))

            logger.info("\nEXAMPLE QUERIES:")
            for i, query in enumerate(templates.get("example_queries", [])):
                logger.info("\nQuery %s:\n%s", i + 1, query.get("query", ""))

    return final_state


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run Tabular to Neo4j converter.")
    parser.add_argument("csv_file", help="Path to the CSV file to analyze")
    parser.add_argument("--output", "-o", help="Path to save the results")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--save-node-outputs",
        "-s",
        action="store_true",
        help="Save the output of each node to files",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default="samples",
        help="Directory to save node outputs to (default: samples)",
    )

    args = parser.parse_args()

    try:
        run_analysis(
            args.csv_file,
            args.output,
            args.verbose,
            args.save_node_outputs,
            args.output_dir,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


def wrap_node_with_output_saving(node_name: str, node_func, node_order: int = 0):
    """
    Wrap a node function with output saving functionality.

    Args:
        node_name: Name of the node
        node_func: Node function to wrap
        node_order: Order of the node in the pipeline (for file naming)

    Returns:
        Wrapped node function
    """

    def wrapped_node(state, config=None):
        # Call the original node function with the config parameter
        result = node_func(state, config) if config is not None else node_func(state)

        # Save the output if output saver is initialized
        output_saver = get_output_saver()
        if output_saver:
            output_saver.save_node_output(node_name, result, node_order)

        return result

    return wrapped_node


if __name__ == "__main__":
    main()
