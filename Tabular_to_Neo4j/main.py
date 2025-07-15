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
    
    # Create graph object only for the original pipelines; the experiment pipelines build graphs internally.
    graph = None
    if pipeline in {"single_table_graph", "multi_table_graph"}:
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
        from Tabular_to_Neo4j.graphs.multi_table_graph import run_multi_table_pipeline
        table_folder = input_path
        try:
            final_state = run_multi_table_pipeline(table_folder)
        except Exception as e:
            raise e
        return final_state
    elif pipeline == "experiments_with_analytics":
        from Tabular_to_Neo4j.graphs.experiments_with_analytics.intra_table_graph_column_map import run_column_map_multi_table_pipeline as run_exp
        table_folder = input_path
        final_state = run_exp(table_folder, use_analytics=True)
        try:
            from pathlib import Path
            import json
            out_path = Path(table_folder) / "GLOBAL" / "final_state.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_state, f, indent=2, default=str)
        except Exception as e:
            print(f"[main] Warning: could not write final_state.json: {e}")
        return final_state
    elif pipeline == "experiments_with_contextualized_analytics":
        from Tabular_to_Neo4j.graphs.experiments_with_contextualized_analytics.intra_table_graph_column_map import run_column_map_multi_table_pipeline as run_exp
        table_folder = input_path
        final_state = run_exp(table_folder, use_analytics=True)
        try:
            from pathlib import Path
            import json
            out_path = Path(table_folder) / "GLOBAL" / "final_state.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_state, f, indent=2, default=str)
        except Exception as e:
            print(f"[main] Warning: could not write final_state.json: {e}")
        return final_state
    elif pipeline == "experiments_without_analytics":
        from Tabular_to_Neo4j.graphs.experiments_without_analytics.intra_table_graph_column_map import run_column_map_multi_table_pipeline as run_exp
        table_folder = input_path
        final_state = run_exp(table_folder, use_analytics=False)
        try:
            from pathlib import Path
            import json
            out_path = Path(table_folder) / "GLOBAL" / "final_state.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_state, f, indent=2, default=str)
        except Exception as e:
            print(f"[main] Warning: could not write final_state.json: {e}")
        return final_state
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run Tabular to Neo4j converter.")
    parser.add_argument("--input_path", help="Path to the CSV file (single-table) or directory of CSVs (multi-table)")
    parser.add_argument("--pipeline", default="single_table_graph", choices=[
        "single_table_graph",
        "multi_table_graph",
        "experiments_with_analytics",
        "experiments_with_contextualized_analytics",
        "experiments_without_analytics",
    ], help="Pipeline graph to run")
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
