from typing import Any
import logging
import os
from langgraph.graph import StateGraph
logger = logging.getLogger(__name__)

def save_results(final_state: Any, output_file: str):
    """
    Save results to a file for both single-table and multi-table outputs.
    """
    if not output_file:
        return
    try:
        logger.info(f"Saving results to {output_file}")
        with open(output_file, "w") as f:
            if isinstance(final_state, dict) and all(isinstance(v, dict) for v in final_state.values()):
                # Multi-table output
                for table_name, table_state in final_state.items():
                    f.write(f"\n=== Table: {table_name} ===\n")
                    if table_state.get("cypher_query_templates"):
                        templates = table_state["cypher_query_templates"]
                        f.write("ENTITY CREATION QUERIES:\n")
                        for i, query in enumerate(templates.get("entity_creation_queries", [])):
                            f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                        f.write("\nRELATIONSHIP QUERIES:\n")
                        for i, query in enumerate(templates.get("relationship_queries", [])):
                            f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                        f.write("\nEXAMPLE QUERIES:\n")
                        for i, query in enumerate(templates.get("example_queries", [])):
                            f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                    else:
                        f.write("No Cypher templates generated.\n")
                    if table_state.get("error_messages"):
                        f.write("\nErrors/Warnings:\n")
                        for error in table_state.get("error_messages", []):
                            f.write(f"  - {error}\n")
            else:
                # Single-table output
                if final_state.get("cypher_query_templates"):
                    templates = final_state["cypher_query_templates"]
                    f.write("ENTITY CREATION QUERIES:\n")
                    for i, query in enumerate(templates.get("entity_creation_queries", [])):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                    f.write("\nRELATIONSHIP QUERIES:\n")
                    for i, query in enumerate(templates.get("relationship_queries", [])):
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

def display_results(final_state: Any, verbose: bool):
    """
    Display results for both single-table and multi-table outputs.
    """
    if not verbose:
        return
    logger.info("Displaying analysis results")
    if isinstance(final_state, dict) and all(isinstance(v, dict) for v in final_state.values()):
        # Multi-table output
        for table_name, table_state in final_state.items():
            logger.info(f"\n=== Table: {table_name} ===")
            if table_state.get("error_messages"):
                logger.info("\nErrors/Warnings encountered during analysis:")
                for error in table_state["error_messages"]:
                    logger.info("  - %s", error)
            if table_state.get("cypher_query_templates"):
                templates = table_state["cypher_query_templates"]
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
    else:
        # Single-table output
        if final_state.get("error_messages"):
            logger.info("\nErrors/Warnings encountered during analysis:")
            for error in final_state["error_messages"]:
                logger.info("  - %s", error)
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

def create_graph(pipeline: str = "single_table_graph") -> 'StateGraph':
    """
    Selects the appropriate workflow graph based on the pipeline argument.
    Only used for single-table pipeline.
    """
    if pipeline == "single_table_graph":
        from Tabular_to_Neo4j.graphs.single_table_graph import create_single_table_graph
        return create_single_table_graph()
    elif pipeline == "multi_table_graph":
        from Tabular_to_Neo4j.graphs.multi_table_graph import create_multi_table_graph
        return create_multi_table_graph()
    else:
        logger.warning(f"Unknown pipeline '{pipeline}', defaulting to single_table_graph.")
        from Tabular_to_Neo4j.graphs.single_table_graph import create_single_table_graph
        return create_single_table_graph()

def validate_input_path(input_path: str, pipeline: str):
    if pipeline == "single_table_graph":
        if not os.path.isfile(input_path):
            logger.error(f"Expected a CSV file for single-table pipeline, got: {input_path}")
            raise ValueError("Input must be a CSV file for single-table pipeline.")
    elif pipeline == "multi_table_graph":
        if not os.path.isdir(input_path):
            logger.error(f"Expected a directory of CSVs for multi-table pipeline, got: {input_path}")
            raise ValueError("Input must be a directory for multi-table pipeline.")
    else:
        logger.error(f"Unknown pipeline: {pipeline}")
        raise ValueError(f"Unknown pipeline: {pipeline}")