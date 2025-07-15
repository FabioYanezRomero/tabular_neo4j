#!/usr/bin/env python
"""Convenience wrapper to launch the *experiments_without_analytics* pipeline.

This script mirrors ``scripts/run_experiment_with_analytics.py`` but runs the
pipeline variant that *does not* calculate full column analytics. This results
in faster execution and lower LLM token usage, at the expense of slightly less
contextual information for reasoning.

Example
-------
python scripts/run_experiment_without_analytics.py \
    --input_path /app/datasets/csvs/diginetica \
    --output_dir samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the main run helper from the package
try:
    from Tabular_to_Neo4j.main import run as run_pipeline
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"Failed to import Tabular_to_Neo4j: {exc}\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run the experiments_without_analytics multi-table pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        default="/app/datasets/csvs/diginetica",
        help="Root directory containing CSV datasets (can have multiple sub-folders).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="samples",
        help="Directory where all pipeline outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401 (no imperative mood requirement)
    """Entry-point used when the module is executed as a script."""
    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Execute the pipeline via the shared `run` helper to ensure identical logic
    run_pipeline(
        input_path=str(input_path),
        output_dir=str(output_dir),
        pipeline="experiments_without_analytics",
    )


if __name__ == "__main__":
    main()
