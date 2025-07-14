#!/usr/bin/env python3
"""Dask-based precompute per-column analytics for all CSV datasets.

Uses out-of-core processing to avoid loading entire tables into memory.
"""
from __future__ import annotations

import json
from pathlib import Path
import csv
import random

from typing import Dict, Any

import dask.dataframe as dd
import pandas as pd
from pandas.errors import ParserError, DtypeWarning

import warnings
from Tabular_to_Neo4j.utils.llm_api import call_llm_api
# suppress mixed-type column warnings
warnings.filterwarnings("ignore", category=DtypeWarning)

from Tabular_to_Neo4j.utils.analytics_utils import analyze_all_columns
from Tabular_to_Neo4j.config.settings import (
    LLM_CONFIGS,
    MAX_SAMPLE_ROWS
)
from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

# DeepJoin contextualized analytics output directory
CONTEXT_OUTPUT_DIR = Path("/app/contextualized_analytics")


def _llm_generate_column_description(table_name: str, column_name: str, samples: list[str], stats: dict[str, Any]) -> dict[str, Any]:
    """Generate semantic analysis for a column using configured LLM."""
    # Prepare analytics for prompt
    # Prepare analytics for prompt
    stats_for_prompt = {k: v for k, v in stats.items() if k not in {"contextual_description"}}
    # Determine data type for prompt customization
    data_type = stats.get("inferred_type") or stats.get("semantic_type") or "unknown"
    dt = data_type.lower()
    if dt in ("int", "integer", "numeric", "float", "int64"):  # Numeric types
        type_instr = "Focus on distribution, range, mean, and outliers."
    elif dt in ("datetime", "date", "timestamp", "datetime64[ns]"):  # Temporal types
        type_instr = "Focus on temporal patterns: earliest, latest, and frequency."
    elif dt in ("categorical", "category"):  # Categorical types
        type_instr = "Focus on category frequency, cardinality, and common values."
    else:  # Text and other types
        type_instr = "Focus on text characteristics: average length, unique values overview, and sample texts."
    # Build prompt including type instruction
    prompt = (
        f"Column data type: {data_type}. {type_instr}\n\n"
        "You are an expert data analyst. Using the following analytics about a column, "
        "return only the JSON object exactly as defined in the schema.\n\n"
        f"Table: {table_name}\n"
        f"Column: {column_name}\n"
        f"Analytics: {json.dumps(stats_for_prompt, ensure_ascii=False)}\n\n"
        "Schema: " + json.dumps(LLM_CONFIGS["analyse"]["output_format"], ensure_ascii=False)
    )
    try:
        # Dispatch to LLM API (Ollama or LMStudio) via unified client
        response_text = call_llm_api(prompt, config=LLM_CONFIGS["analyse"])
        result = json.loads(response_text)
    except Exception as exc:
        logger.warning("LLM semantic analysis failed for %s.%s: %s", table_name, column_name, exc)
        result = {"semantic_type": "Unknown", "contextual_description": ""}
    return result



def _write_contextualized_text(table_name: str, column_name: str, enriched: dict[str, Any]) -> None:
    """Write a plain-text contextualization suitable as LM input."""
    out_dir = CONTEXT_OUTPUT_DIR / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{column_name}.txt"
    samples = ", ".join(map(str, enriched.get("sampled_values", [])))
    lines = [
        f"Table: {table_name}",
        f"Column: {column_name}",
        f"Type: {enriched.get('semantic_type', enriched.get('inferred_type', 'unknown'))}",
        f"Context: {enriched.get('contextual_description', '')}",
        f"Stats: distinct={enriched.get('distinct_count')}, min_len={enriched.get('min_length')}, max_len={enriched.get('max_length')} ",
        f"Min value: {enriched.get('min_value')}",
        f"Max value: {enriched.get('max_value')}",
        f"Samples: {samples}",
    ]
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as exc:
        logger.error("Failed writing contextualized text for %s.%s: %s", table_name, column_name, exc)

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
    # Table-specific dtype overrides
    table_name = csv_path.stem
    dtype_map: Dict[str, str] = {}
    # movielens user file has alphanumeric zip codes
    if table_name == 'u_user':
        dtype_map['zip_code'] = 'object'
    try:
        # Read CSV lazily with 64MB partitions
        ddf = dd.read_csv(
            str(csv_path),
            sep=sep,
            assume_missing=True,
            blocksize="64MB",
            low_memory=False,
            dtype=dtype_map,
        )  # type: ignore
        cols = list(ddf.columns)
    except ParserError as e:
        logger.warning("ParserError reading %s: %s; falling back to pandas with skip", csv_path, e)
        try:
            df = pd.read_csv(
                str(csv_path),
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                low_memory=False,
                dtype=dtype_map,
            )  # type: ignore
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
            # Compute single-column pandas DataFrame for this column only
            pdf = ddf[[col]].compute()

            # Auto-convert object dtype columns to float if >90% numeric
            if pdf[col].dtype == object:
                try:
                    converted = pd.to_numeric(pdf[col], errors="coerce")  # type: ignore[arg-type]
                    num_numeric = converted.dropna().shape[0]  # type: ignore[attr-defined]
                    total = pdf.shape[0]
                    if total > 0 and num_numeric / total > 0.9:
                        pdf[col] = converted
                except Exception:
                    pass
            # Apply dtype override if specified
            if col in dtype_map:
                try:
                    pdf[col] = pdf[col].astype(dtype_map[col])
                except Exception as cast_exc:
                    logger.warning("Casting error for %s in %s: %s", col, csv_path, cast_exc)

            # Base statistics using helper
            stats_dict: Dict[str, Any] = analyze_all_columns(pdf).get(col, {})
            # Ensure inferred type and reuse analytics from analyze_all_columns
            stats_dict["inferred_type"] = stats_dict.get("data_type", "unknown")
            # Samples – up to 5 randomly chosen distinct non-null values
            unique_vals = pdf[col].dropna().unique()
            sample_size = min(MAX_SAMPLE_ROWS, len(unique_vals))
            sampled_values = random.sample(list(unique_vals), sample_size) if sample_size > 0 else []
            stats_dict["sampled_values"] = sampled_values
            # Distinct count convenience
            stats_dict["distinct_count"] = int(pdf[col].nunique(dropna=True))
            # Add table + column metadata
            stats_dict["table_name"] = table_name
            stats_dict["column_name"] = col
            # Semantic LLM analysis
            semantic_info = _llm_generate_column_description(table_name, col, sampled_values, stats_dict)
            stats_dict.update(semantic_info)
            analytics[col] = stats_dict
            # Write contextualized text for LLM input
            _write_contextualized_text(table_name, col, stats_dict)
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
