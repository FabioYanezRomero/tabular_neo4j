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
# Directory for samples-only outputs
NO_ANALYTICS_DIR = Path("/app/no_analytics")

def contextualize_integer_float(token_sequence: str, column_info: dict) -> str:
    """
    Contextualize integer/float token sequences using JSON metadata.
    Implementation provided by the user.
    """
    delimiter = column_info.get('list_delimiter', ',')
    tokens = token_sequence.split(delimiter) if token_sequence else []
    token_count = len(tokens)
    unique_tokens = len(set(tokens)) if tokens else 0
    repeated_tokens = token_count - unique_tokens

    token_list = ", ".join(tokens)

    context = (
        f"{column_info['column_name']}: {token_list}. "

        f"This {column_info['cardinality_type']} cardinality {column_info['data_type']} sequence has {column_info['distinct_count']} distinct combinations "
        f"from {column_info['total_count']} total entries. "
        f"Token value range: {column_info['min_value']:.0f} to {column_info['max_value']:.0f}, "
        f"average value: {column_info['avg_value']:.0f}. "
        f"Sequence length statistics: average {column_info.get('tokens_per_cell_avg', 1):.1f}, "
        f"range {column_info.get('tokens_per_cell_min', 1)}-{column_info.get('tokens_per_cell_max', 1)} tokens. "
        f"Distribution uniformity: {column_info['distribution']['uniformity']*100:.1f}%, "
        f"uniqueness ratio: {column_info['uniqueness_ratio']*100:.2f}%. "
        f"Most frequent token: {column_info['mode_values'][0]}. "
        f"Table: {column_info['table_name']}. "
        f"Context: {column_info['contextual_description']}"
    )
    return context


def contextualize_string(string_value: str, column_info: dict) -> str:
    """Contextualize string values using JSON metadata"""
    value_length = len(string_value)
    is_mode_value = string_value in column_info['mode_values']
    popularity = "mode" if is_mode_value else "non-mode"

    context = (
        f"{column_info['column_name']}: {string_value} (length: {value_length} chars, {popularity} value). "
        f"String statistics: min {column_info['value_lengths']['min']}, "
        f"max {column_info['value_lengths']['max']}, "
        f"mean {column_info['value_lengths']['mean']:.1f}, "
        f"median {column_info['value_lengths']['median']}, "
        f"std {column_info['value_lengths']['std']:.2f}. "
        f"Cardinality: {column_info['cardinality_type']} ({column_info['distinct_count']} distinct values). "
        f"Distribution: uniformity {column_info['distribution']['uniformity']*100:.1f}%, "
        f"top frequency {column_info['distribution']['top_value_frequency']*100:.2f}%. "
        f"Uniqueness ratio: {column_info['uniqueness_ratio']:.3f}. "
        f"Table: {column_info['table_name']} ({column_info['total_count']} entries). "
        f"Context: {column_info['contextual_description']}"
    )
    return context


def contextualize_boolean(bool_value: bool, column_info: dict) -> str:
    """Contextualize boolean values using JSON metadata"""
    status = "enabled" if bool_value else "disabled"
    mode_value = column_info['mode_values'][0]
    is_majority = bool_value == mode_value
    case_type = "majority" if is_majority else "minority"

    context = (
        f"{column_info['column_name']}: {bool_value} ({status}, {case_type} case). "
        f"Data type: {column_info['data_type']} with {column_info['cardinality_type']} cardinality. "
        f"Distribution: top value frequency {column_info['distribution']['top_value_frequency']*100:.1f}%, "
        f"uniformity {column_info['distribution']['uniformity']*100:.1f}%. "
        f"Mode value: {mode_value}. "
        f"Uniqueness ratio: {column_info['uniqueness_ratio']:.9f}. "
        f"Total entries: {column_info['total_count']:,}, distinct values: {column_info['distinct_count']}. "
        f"Table: {column_info['table_name']}. "
        f"Context: {column_info['contextual_description']}"
    )
    return context


def contextualize_date(date_value: str, column_info: dict) -> str:
    """Contextualize date values using JSON metadata"""
    from datetime import datetime
    date_obj = datetime.strptime(date_value, "%Y-%m-%d")
    day_of_week = date_obj.strftime("%A")
    month_name = date_obj.strftime("%B")

    context = (
        f"{column_info['column_name']}: {date_value} ({day_of_week}, {month_name} {date_obj.day}, {date_obj.year}). "
        f"Date range: {column_info['min_value']} to {column_info['max_value']}. "
        f"Cardinality: {column_info['cardinality_type']} ({column_info['distinct_count']} unique dates). "
        f"Distribution: uniformity {column_info['distribution']['uniformity']*100:.2f}%, "
        f"top frequency {column_info['distribution']['top_value_frequency']*100:.2f}%. "
        f"Most frequent: {column_info['mode_values'][0]}. "
        f"Uniqueness ratio: {column_info['uniqueness_ratio']:.8f}. "
        f"Total events: {column_info['total_count']:,} in table {column_info['table_name']}. "
        f"Context: {column_info['contextual_description']}"
    )
    return context

import numpy as np

def _to_py_scalar(val):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _make_json_serializable(obj):  # type: ignore[override]
    """Recursively convert numpy types in a nested structure to Python primitives."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    return _to_py_scalar(obj)


def _llm_generate_column_description(table_name: str, column_name: str, samples: list[str], stats: dict[str, Any]) -> str:
    """Generate semantic analysis for a column using configured LLM."""
    # Prepare analytics for prompt
    stats_for_prompt = {k: v for k, v in stats.items() if k not in {"contextual_description"}}
    # Determine data type for prompt customization
    data_type = stats.get("inferred_type") or stats.get("semantic_type") or "unknown"
    dt = data_type.lower()
    # Error on undetected (unknown) types
    if dt == "unknown":
        logger.error("Data type detection failed for %s.%s: %s", table_name, column_name, stats)
        raise ValueError(f"Unknown data type for {table_name}.{column_name}")
    stats_for_prompt['data_type'] = dt
    stats_for_prompt['sampled_values'] = samples
    if dt in ("int", "integer", "numeric", "float", "int64"):  # Numeric types
        type_instr = "Focus on distribution, range, mean, and outliers."
    elif dt in ("datetime", "date", "timestamp", "datetime64[ns]"):  # Temporal types
        type_instr = "Focus on temporal patterns: earliest, latest, and frequency."
    elif dt in ("categorical", "category"):  # Categorical types
        type_instr = "Focus on category frequency, cardinality, and common values."
    else:  # Text and other types
        type_instr = "Focus on text characteristics: average length, unique values overview, and sample texts."
    # Ensure JSON-serializable analytics
    stats_for_prompt = _make_json_serializable(stats_for_prompt)  # type: ignore[name-defined]
    # Build prompt including type instruction
    prompt = (  # use dt for data type label

        f"Column data type: {dt}. {type_instr}\n\n"
        "You are an expert data analyst. Using the following analytics about a column, "
        "write a concise natural-language description of what this column represents.\n\n"
        f"Table: {table_name}\n"
        f"Column: {column_name}\n"
        f"Analytics: {json.dumps(stats_for_prompt, ensure_ascii=False)}\n\n"
        "IMPORTANT: Respond ONLY with plain text – no markdown, no JSON, no code fences. Return just the description sentence(s)."
    )
    try:
        # Dispatch to LLM API (Ollama or LMStudio) via unified client
        response_text = call_llm_api(prompt, config=LLM_CONFIGS["analyse"])
        description = response_text.strip().strip('`').strip()
        # remove leading/lowering quotes if model wrapped
        description = description.strip('"')
        result = description
    except Exception as exc:
        logger.warning("LLM semantic analysis failed for %s.%s: %s", table_name, column_name, exc)
        result = ""
    return result  # description string (may be empty on failure)


def _write_samples_only(dataset_name: str, table_name: str, column_name: str, samples: list[Any]) -> None:
    """Write only the sampled values for a column (no analytics)."""
    out_dir = NO_ANALYTICS_DIR / dataset_name / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{column_name}.txt"
    try:
        path.write_text(", ".join(map(str, samples)), encoding="utf-8")
    except Exception as exc:
        logger.error("Failed writing samples for %s.%s: %s", table_name, column_name, exc)


def _write_contextualized_text(dataset_name: str, table_name: str, column_name: str, enriched: dict[str, Any]) -> None:
    """Write a plain-text contextualization using specialized helper functions."""
    out_dir = CONTEXT_OUTPUT_DIR / dataset_name / table_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{column_name}.txt"

    data_type = (enriched.get("data_type") or enriched.get("semantic_type") or enriched.get("inferred_type") or "unknown").lower()
    context_text: str

    try:
        # Choose contextualization strategy by data type
        if data_type in {"integer", "float"}:
            # Build a representative token sequence (use sampled values if available)
            # Ensure list_delimiter exists for helper
            delimiter = enriched.get("list_delimiter", ",")
            if "list_delimiter" not in enriched:
                enriched["list_delimiter"] = delimiter
            token_sequence = delimiter.join(map(str, enriched.get("sampled_values", [])[:10]))
            context_text = contextualize_integer_float(token_sequence, enriched)
        elif data_type == "string":
            sample_val = str(enriched.get("sampled_values", [""])[0]) if enriched.get("sampled_values") else ""
            context_text = contextualize_string(sample_val, enriched)
        elif data_type in {"boolean", "categorical"}:
            mode_val = enriched.get("mode_values", [True])[0]
            # Convert possible string representation to boolean
            if isinstance(mode_val, str):
                mode_val_bool = mode_val.lower() in {"true", "1", "yes", "y", "t"}
            else:
                mode_val_bool = bool(mode_val)
            context_text = contextualize_boolean(mode_val_bool, enriched)
        elif data_type in {"date", "datetime"}:
            date_val = str(enriched.get("mode_values", [enriched.get("min_value", "1970-01-01")])[0])
            context_text = contextualize_date(date_val, enriched)
        else:
            # Fallback to LLM-generated contextual description if available
            context_text = enriched.get("contextual_description", "")
    except Exception as exc:
        logger.error("Contextualization helper failed for %s.%s: %s", table_name, column_name, exc)
        context_text = enriched.get("contextual_description", "")

    try:
        path.write_text(context_text, encoding="utf-8")
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
                    total = pdf.shape[0]
                    num_numeric = converted.dropna().shape[0]  # type: ignore[attr-defined]
                    if total > 0 and num_numeric / total > 0.9:
                        pdf[col] = converted
                except Exception:
                    pass
            # If object, attempt datetime conversion if >90% parseable
            if pdf[col].dtype == object:
                try:
                    dt_parsed = pd.to_datetime(pdf[col], errors="coerce", utc=False, infer_datetime_format=True)
                    num_dt = dt_parsed.dropna().shape[0]
                    if total > 0 and num_dt / total > 0.9:
                        pdf[col] = dt_parsed
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
            description = _llm_generate_column_description(table_name, col, sampled_values, stats_dict)
            stats_dict["contextual_description"] = description
            # ensure serializable early
            stats_dict = _make_json_serializable(stats_dict)
            analytics[col] = stats_dict
            # Write contextualized text for LLM input
            _write_contextualized_text(dataset_name, table_name, col, stats_dict)
            # Also write samples-only file in /app/no_analytics
            _write_samples_only(dataset_name, table_name, col, sampled_values)
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
                json.dump(_make_json_serializable(stats), f, ensure_ascii=False, indent=2)
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
