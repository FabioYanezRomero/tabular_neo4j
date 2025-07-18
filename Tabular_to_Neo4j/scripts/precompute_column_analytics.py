"""Dask-based precompute per-column analytics for all CSV datasets.

Uses out-of-core processing to avoid loading entire tables into memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import dask.dataframe as dd
import pandas as pd
from pandas.errors import ParserError
from tqdm import tqdm

from Tabular_to_Neo4j.utils.analytics_utils import analyze_all_columns
from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

# Adjust these paths as needed
DATASETS_DIR = Path("/app/datasets/csvs")
OUTPUT_DIR = Path("/app/analytics/csvs")


def compute_and_save_for_csv(csv_path: Path, dataset_name: str) -> None:
    """Compute and save analytics for a single CSV using Dask."""
    try:
        # Read CSV lazily with 64MB partitions
        ddf = dd.read_csv(csv_path, assume_missing=True, blocksize="64MB")
        cols = list(ddf.columns)
    except ParserError as e:
        logger.warning("ParserError reading %s: %s; falling back to pandas with skip", csv_path, e)
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            ddf = dd.from_pandas(df, npartitions=1)
            cols = list(df.columns)
        except Exception as e2:
            logger.error("Failed pandas fallback reading %s: %s", csv_path, e2)
            return
    except Exception as e:
        logger.error("Failed reading %s: %s", csv_path, e)
        return

    analytics: Dict[str, Any] = {}
    for col in cols:
        try:
            # Compute single-column pandas DataFrame
            pdf = ddf[[col]].compute()
            stats = analyze_all_columns(pdf)
            analytics[col] = stats.get(col, {})
        except Exception as e:
            logger.error("Failed analytics for %s column %s: %s", csv_path, col, e)

    # Persist each column's stats
    table_name = csv_path.stem
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

    logger.info("Processed %d columns for %s", len(analytics), csv_path)


def main() -> None:
    # Verify datasets directory
    if not DATASETS_DIR.exists():
        logger.error("Datasets directory %s not found", DATASETS_DIR)
        return

    # Dataset subdirectories
    dataset_dirs = [d for d in DATASETS_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        logger.warning("No dataset directories found under %s", DATASETS_DIR)
        return

    # Iterate per-dataset with progress
    for dataset_dir in tqdm(dataset_dirs, desc="Datasets"):
        dataset_name = dataset_dir.name
        csv_files = list(dataset_dir.rglob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found under %s", dataset_dir)
            continue

        # Iterate per-table with nested progress
        for csv_path in tqdm(csv_files, desc=f"Tables in {dataset_name}", leave=False):
            compute_and_save_for_csv(csv_path, dataset_name)

    logger.info("Analytics precomputation completed.")


if __name__ == "__main__":
    main()
