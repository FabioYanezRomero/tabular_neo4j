"""
Main script for the Tabular to Neo4j converter using LangGraph.
"""

import argparse
import os
import time
from typing import Dict, Any
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.utils.logging_config import get_logger, setup_logging
import sys

from Tabular_to_Neo4j.utils.result_utils import save_results, display_results, validate_input_path, create_graph
from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver
from Tabular_to_Neo4j.utils.logging_config import set_log_file_path

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

def run(
    input_path: str,
    output_file: str = None,
    verbose: bool = False,
    save_node_outputs: bool = False,
    output_dir: str = "samples",
    pipeline: str = "single_table_graph",
) -> Dict[str, Any]:
    """
    Run the CSV analysis and Neo4j schema inference for single-table or multi-table pipelines.
    Args:
        input_path: Path to the CSV file (single-table) or directory of CSVs (multi-table)
        output_file: Optional path to save the results (single-table: one file, multi-table: summary)
        verbose: Whether to print verbose output
    Returns:
        The final graph state (single-table) or MultiTableGraphState (multi-table)
    """
    
    validate_input_path(input_path, pipeline)
    logger.info(f"Starting analysis with pipeline: {pipeline}")
    
    # Always initialize output_saver and propagate timestamp for prompt/LLM output saving
    output_saver = initialize_output_saver(output_dir)
    
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized. All output saving must use the same timestamp for the run.")
    if save_node_outputs:
        logs_dir = os.path.join(output_saver.base_dir, output_saver.timestamp, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, "pipeline.log")
        set_log_file_path(log_file_path)
    
    # Abstract graph creation
    graph = create_graph(pipeline)
    start_time = time.time()
    if pipeline == "single_table_graph":
        from Tabular_to_Neo4j.graphs.single_table_graph import run_pipeline
        csv_file_path = input_path
        final_state = run_pipeline(graph, csv_file_path)
        save_results(final_state, output_file)
        display_results(final_state, verbose)
        # Save the final state as metadata
        import json
        from pathlib import Path
        meta_dir = Path(output_saver.base_dir) / output_saver.timestamp
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "final_state.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2, default=str)
        return final_state
    
    elif pipeline == "multi_table_graph":
        from Tabular_to_Neo4j.graphs.multi_table_graph import (
            initialize_multi_table_state, run_multi_table_pipeline
        )
        table_folder = input_path
        logger.info(f"Initializing multi-table state from directory: {table_folder}")
        state = initialize_multi_table_state(table_folder)
        logger.info(f"Running multi-table pipeline on {len(state)} tables: {list(state.keys())}")
        try:
            final_state = run_multi_table_pipeline(state)
            execution_time = time.time() - start_time
            logger.info(f"Multi-table analysis completed in {execution_time:.2f} seconds")
            for table_name, table_state in final_state.items():
                logger.info(f"\n=== Table: {table_name} ===")
                if table_state.get("error_messages"):
                    logger.info("\nErrors/Warnings encountered during analysis:")
                    for error in table_state.get("error_messages", []):
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
        except Exception as e:
            logger.error(f"Error processing multi-table pipeline: {e}")
            raise
        return final_state
    else:
        logger.error(f"Unknown pipeline: {pipeline}")
        raise ValueError(f"Unknown pipeline: {pipeline}")


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run Tabular to Neo4j converter.")
    parser.add_argument("--input_path", help="Path to the CSV file (single-table) or directory of CSVs (multi-table)")
    parser.add_argument("--pipeline", default="single_table_graph", choices=["single_table_graph", "multi_table_graph"], help="Pipeline/graph to use: 'single_table_graph' (default) or 'multi_table_graph'")
    parser.add_argument("--output", "-o", help="Path to save the results (single-table: one file, multi-table: summary)")
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

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

    try:
        run(
            args.input_path,
            args.output,
            args.verbose,
            args.save_node_outputs,
            args.output_dir,
            args.pipeline,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
