#!/usr/bin/env python3
"""
Script to run the Tabular to Neo4j converter with Ollama integration.
"""

import argparse
import os
import sys
from pathlib import Path

from Tabular_to_Neo4j.utils.logging_config import get_logger, setup_logging
from Tabular_to_Neo4j.utils.llm_api import load_llm_for_state

logger = get_logger(__name__)

# Add the repository root to the Python path if needed
repo_root = Path(__file__).parent.absolute()
if str(repo_root.parent) not in sys.path:
    sys.path.insert(0, str(repo_root.parent))


def main():
    """
    Main entry point for running with Ollama integration.
    """
    parser = argparse.ArgumentParser(description='Run Tabular to Neo4j converter with Ollama integration.')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('--output', '-o', help='Path to save the results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--retries', type=int, default=3, help='Number of connection retries')
    parser.add_argument('--save-node-outputs', '-s', action='store_true', help='Save the output of each node to files')
    parser.add_argument('--output-dir', '-d', default="samples", help='Directory to save node outputs to (default: samples)')
    args = parser.parse_args()

    setup_logging()

    from Tabular_to_Neo4j.main import run_analysis

    try:
        final_state = run_analysis(
            csv_file_path=args.csv_file,
            output_file=args.output,
            verbose=args.verbose,
            save_node_outputs=args.save_node_outputs,
            output_dir=args.output_dir,
        )
        if final_state and 'cypher_query_templates' in final_state:
            templates = final_state['cypher_query_templates']
            entity_count = len(templates.get('entity_creation_queries', []))
            relationship_count = len(templates.get('relationship_queries', []))
            logger.info(
                "Generated %s entity creation queries and %s relationship queries.",
                entity_count,
                relationship_count,
            )
        else:
            logger.warning("⚠️ No Cypher templates were generated.")
    except Exception as e:
        logger.error("❌ Error during analysis: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
