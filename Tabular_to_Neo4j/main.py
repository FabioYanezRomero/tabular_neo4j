"""
Main script for the Tabular to Neo4j converter using LangGraph.
"""

import argparse
from typing import Dict, Any
from Tabular_to_Neo4j.utils.logging_config import get_logger, setup_logging
import sys

from Tabular_to_Neo4j.utils.result_utils import validate_input_path, create_graph
from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver

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
    output_dir: str = "samples",
    pipeline: str = "single_table_graph",
) -> Dict[str, Any]:
    """
    Run the CSV analysis and Neo4j schema inference for single-table or multi-table pipelines.
    Args:
        input_path: Path to the CSV file (single-table) or directory of CSVs (multi-table)
        output_dir: Directory to save outputs to (default: samples)
        pipeline: Pipeline to use: 'single_table_graph' (default) or 'multi_table_graph'
    Returns:
        The final graph state (single-table) or MultiTableGraphState (multi-table)
    """
    
    validate_input_path(input_path, pipeline)
    logger.info(f"Starting analysis with pipeline: {pipeline}")
    
    # Always initialize output_saver and propagate timestamp for prompt/LLM output saving
    output_saver = initialize_output_saver(output_dir)
    
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized. All output saving must use the same timestamp for the run.")
    
    # Abstract graph creation
    graph = create_graph(pipeline)
    if pipeline == "single_table_graph":
        from Tabular_to_Neo4j.graphs.single_table_graph import run_pipeline
        csv_file_path = input_path
        final_state = run_pipeline(graph, csv_file_path)
        
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
        state = initialize_multi_table_state(table_folder)
        try:
            final_state = run_multi_table_pipeline(state)
        except Exception as e:
            raise e
        return final_state
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run Tabular to Neo4j converter.")
    parser.add_argument("--input_path", help="Path to the CSV file (single-table) or directory of CSVs (multi-table)")
    parser.add_argument("--pipeline", default="single_table_graph", choices=["single_table_graph", "multi_table_graph"], help="Pipeline/graph to use: 'single_table_graph' (default) or 'multi_table_graph'")
    parser.add_argument("--output_dir", "-o", default="samples", help="Path to save the results (single-table: one file, multi-table: summary)")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    try:
        run(
            args.input_path,
            args.output_dir,
            args.pipeline,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
