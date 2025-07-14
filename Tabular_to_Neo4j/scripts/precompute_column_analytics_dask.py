"""Dask-based precompute per-column analytics for all CSV datasets.

Uses out-of-core processing to avoid loading entire tables into memory.
"""
from __future__ import annotations

import json
from pathlib import Path
import csv
from typing import Dict, Any

import dask.dataframe as dd
import pandas as pd
from pandas.errors import ParserError

from Tabular_to_Neo4j.utils.analytics_utils import analyze_all_columns
from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

# Adjust these paths as needed
DATASETS_DIR = Path("/app/datasets/csvs")
OUTPUT_DIR = Path("/app/analytics/csvs")


def compute_and_save_for_csv(csv_path: Path, dataset_name: str) -> None:
    # Detect delimiter by header count heuristic
    try:
        header = csv_path.open().readline()
        # Choose among common delimiters by highest count in header
        candidates = [',', ';', '\t', '|']
        sep = max(candidates, key=lambda d: header.count(d))
        if header.count(sep) == 0:
            sep = ','
    except Exception:
        sep = ','
    """Compute and save analytics for a single CSV using Dask."""
    try:
        # Read CSV lazily with 64MB partitions
        ddf = dd.read_csv(csv_path, sep=sep, assume_missing=True, blocksize="64MB", engine="python", on_bad_lines="skip")
        cols = list(ddf.columns)
    except ParserError as e:
        logger.warning("ParserError reading %s: %s; falling back to pandas with skip", csv_path, e)
        try:
            df = pd.read_csv(csv_path, sep=sep, engine="python", on_bad_lines='skip')
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
    if not DATASETS_DIR.exists():
        logger.error("Datasets directory %s not found", DATASETS_DIR)
        return

    csv_files = list(DATASETS_DIR.rglob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found under %s", DATASETS_DIR)
        return

    logger.info("Found %d CSV files – starting analytics precompute", len(csv_files))
    for csv_path in csv_files:
        rel = csv_path.relative_to(DATASETS_DIR)
        dataset_name = rel.parts[0]
        compute_and_save_for_csv(csv_path, dataset_name)

    logger.info("Analytics precomputation completed.")


if __name__ == "__main__":
    main()
