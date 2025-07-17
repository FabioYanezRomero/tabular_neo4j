#!/usr/bin/env python
"""Convenience wrapper to launch the *experiments_with_contextualized_analytics* pipeline.

This script parallels the *with analytics* and *without analytics* runners but
executes the pipeline variant that uses column analytics contextualized within
each prompt (providing richer, focused signals to the LLM).

Example
-------
python scripts/run_experiment_with_contextualized_analytics.py \
    --input_path /app/datasets/csvs/diginetica \
    --output_dir samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from Tabular_to_Neo4j.main import run as run_pipeline
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"Failed to import Tabular_to_Neo4j: {exc}\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run the experiments_with_contextualized_analytics pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        default="/app/datasets/csvs/movielens_100k",
        help="Root directory containing CSV datasets (may include sub-folders).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="samples",
        help="Directory where all pipeline outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401
    """Entry point when executed as a script."""
    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for i in range(10):
        run_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            pipeline="experiments_with_contextualized_analytics",
        )

if __name__ == "__main__":
    main()
