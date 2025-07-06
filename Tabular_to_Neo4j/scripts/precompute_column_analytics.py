"""Precompute per-column analytics for all CSV datasets.

This script scans `/app/datasets/csvs/<dataset_name>/**/*.csv`, computes the
analytics for every column using `analytics_utils.analyze_all_columns`, and
stores the results under `/app/analytics/csvs/<dataset_name>/<table>/<column>.json`.

It is intended to be executed *once* (or whenever datasets change) so that the
LangGraph pipeline can fetch analytics directly from the filesystem instead of
running the analytics node at runtime.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from Tabular_to_Neo4j.utils.analytics_utils import analyze_all_columns
from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

DATASETS_DIR = Path("/app/datasets/csvs")
OUTPUT_DIR = Path("/app/analytics/csvs")


def compute_and_save_for_csv(csv_path: Path, dataset_name: str) -> None:
    """Compute analytics for a single CSV and persist to OUTPUT_DIR."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed reading %s: %s", csv_path, e)
        return

    analytics = analyze_all_columns(df)

    # Build out path: /app/analytics/csvs/<dataset>/<table>/<column>.json
    table_name = csv_path.stem  # file name without extension
    base_out = OUTPUT_DIR / dataset_name / table_name
    for col, stats in analytics.items():
        out_dir = base_out
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{col}.json"
        try:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed writing analytics for %s %s: %s", table_name, col, e)
    logger.info("Processed %s (%d columns)", csv_path, len(analytics))


def main() -> None:
    if not DATASETS_DIR.exists():
        logger.error("Datasets directory %s not found", DATASETS_DIR)
        return

    csv_files = list(DATASETS_DIR.rglob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found under %s", DATASETS_DIR)
        return

    logger.info("Found %d CSV files – starting analytics precompute", len(csv_files))
    for csv_path in csv_files:
        # Determine dataset name as first directory under DATASETS_DIR
        rel = csv_path.relative_to(DATASETS_DIR)
        dataset_name = rel.parts[0]
        compute_and_save_for_csv(csv_path, dataset_name)

    logger.info("Analytics precomputation completed.")


if __name__ == "__main__":
    main()
