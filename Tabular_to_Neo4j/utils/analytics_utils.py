"""
Utility functions for column analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import re
from datetime import datetime

def calculate_uniqueness_ratio(column: pd.Series) -> float:
    """
    Calculate the uniqueness ratio of a column (distinct values / total values).
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Uniqueness ratio as a float between 0 and 1
    """
    if len(column) == 0:
        return 0.0
    return column.nunique() / len(column)

def calculate_cardinality(column: pd.Series) -> int:
    """
    Calculate the cardinality (number of distinct values) of a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Cardinality as an integer
    """
    return column.nunique()

def calculate_missing_percentage(column: pd.Series) -> float:
    """
    Calculate the percentage of missing values in a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Missing value percentage as a float between 0 and 1
    """
    if len(column) == 0:
        return 0.0
    # Explicitly cast to float to satisfy static type checkers that infer a possible Series return type
    return float(column.isna().mean())

def detect_data_type(column: pd.Series) -> str:
    """
    Detect the predominant data type of a column with more precise categorization.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        String representing the data type ('integer', 'float', 'string', 'date', 'datetime', 'boolean', 'categorical', 'unknown')
    """
    # Clean column by dropping NaN values
    clean_column = column.dropna()
    
    if len(clean_column) == 0:
        return 'unknown'
    
    # Check original pandas dtype first with precise ordering
    orig_dtype = str(column.dtype).lower()
    # Prioritize datetime-related dtypes to avoid false keyword hits (e.g., "datetime64" contains "int")
    if any(x in orig_dtype for x in ('datetime', 'date', 'timedelta')):
        return 'datetime'
    if 'bool' in orig_dtype:
        return 'boolean'
    if orig_dtype.startswith('int') or orig_dtype.endswith('int') or orig_dtype == 'int64':
        # Check if integers may actually be encoded dates in YYYYMMDD format
        sample_ints = clean_column.dropna().astype(str).head(100)
        date_like_count = 0
        for val in sample_ints:
            if len(val) == 8 and val.isdigit():
                try:
                    datetime.strptime(val, "%Y%m%d")
                    date_like_count += 1
                except Exception:
                    pass
        if date_like_count / max(len(sample_ints), 1) > 0.8:
            return 'date'
        return 'integer'
    if 'float' in orig_dtype:
        # Detect floats that are effectively integers (e.g., 1.0, 2.0)
        sample_vals = clean_column.dropna().head(100)
        if not sample_vals.empty and (sample_vals % 1 == 0).mean() > 0.95:  # type: ignore[operator]
            return 'integer'
        return 'float'
    elif 'category' in orig_dtype:
        return 'categorical'
    
    # Detect delimited (multi-value) strings early
    # Use a random sample (up to 500) to avoid bias from early rows lacking delimiters
    sample_size = min(500, len(clean_column))
    str_sample = clean_column.astype(str).sample(n=sample_size, random_state=42, replace=False)
    delimiters = [',', ';', '|', ' ']
    delim_count = sum(any(d in val for d in delimiters) for val in str_sample)
    # Check for list-like structures based on delimiters
    tokens_per_cell_counts = []
    if delim_count > 0:
        # Calculate token counts for rows that have delimiters
        for s in str_sample:
            if any(d in s for d in delimiters):
                tokens_per_cell_counts.append(len([t for t in re.split(r'[ ,;|]+', s) if t]))
    # Decide list status: more than one token in at least 3 rows OR >1% delimiter presence
    if (delim_count / max(len(str_sample), 1) > 0.01) or (sum(c > 1 for c in tokens_per_cell_counts) >= 3):
        # Detect if tokens are mostly numeric to distinguish numeric vs string lists
        tokens: List[str] = []
        for s in str_sample:
            tokens.extend(re.split(r'[ ,;|]+', s))
        tokens = [t for t in tokens if t]
        if tokens:
            numeric_tokens = sum(tok.replace('.', '', 1).isdigit() for tok in tokens)
            if numeric_tokens / len(tokens) > 0.9:
                return 'list_numeric'
        return 'list_string'

    # Try to convert to numeric with coverage check
    try:
        numeric_values = pd.to_numeric(clean_column, errors='coerce')
        numeric_series = numeric_values if isinstance(numeric_values, pd.Series) else pd.Series([numeric_values])
        non_na_numeric = numeric_series.dropna()
        coverage = len(non_na_numeric) / len(clean_column)
        if coverage < 0.9:
            # Not enough numeric coverage – fall through to date/string logic
            raise ValueError("Low numeric coverage")
        # Determine if all numeric values are whole numbers (no fractional part)
        is_integer = non_na_numeric.mod(1).eq(0).all()
        return 'integer' if is_integer else 'float'
    except Exception:
        pass
    
    # Check if all values are boolean
    if set(clean_column.astype(str).str.lower().unique()).issubset({'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}):
        return 'boolean'
    
    # Check if values are categorical (few unique values compared to total)
    unique_ratio = len(clean_column.unique()) / len(clean_column)
    if unique_ratio < 0.1 and len(clean_column.unique()) < 20:
        return 'categorical'
    
    # Check if all values are dates
    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']
    date_count = 0
    datetime_count = 0
    sample_size = min(100, len(clean_column))
    sample = clean_column.astype(str).sample(sample_size)
    
    for val in sample:
        # Check for date formats
        for date_format in date_formats:
            try:
                datetime.strptime(val, date_format)
                date_count += 1
                break
            except:
                pass
        
        # Check for datetime formats
        try:
            # Try common datetime formats
            if ' ' in val and ((':' in val) or ('T' in val)):
                pd.to_datetime(val)
                datetime_count += 1
        except:
            pass
    
    if date_count / sample_size > 0.8:
        return 'date'
    elif datetime_count / sample_size > 0.8:
        return 'datetime'
    
    # Default to string
    return 'string'


def calculate_value_length_stats(column: pd.Series) -> Dict[str, float]:
    """
    Calculate statistics about the length of string values in a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with length statistics (min, max, mean, median)
    """
    # Convert to string and calculate lengths
    clean_column = column.dropna()
    if len(clean_column) == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
    
    # Convert all values to strings and get their lengths
    try:
        lengths = clean_column.astype(str).apply(len)
        return {
            'min': float(lengths.min()),
            'max': float(lengths.max()),
            'mean': float(lengths.mean()),
            'median': float(lengths.median()),
            'std': float(lengths.std()) if len(lengths) > 1 else 0.0
        }
    except Exception as e:
        # Return default values if calculation fails
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}


def analyze_value_distribution(column: pd.Series) -> Dict[str, Any]:
    """
    Analyze the distribution of values in a column.
    
    Args:
        column: Pandas Series representing a column
        
    Returns:
        Dictionary with distribution metrics
    """
    clean_column = column.dropna()
    if len(clean_column) == 0 or clean_column.nunique() <= 1:
        return {'uniformity': 0.0, 'top_value_frequency': 0.0}
    
    # Calculate value frequencies
    value_counts = clean_column.value_counts(normalize=True)
    
    # Uniformity: how evenly distributed the values are (higher = more uniform)
    # Using a simplified entropy calculation
    uniformity = 1.0 - value_counts.max()
    
    # Top value frequency
    top_value_frequency = value_counts.iloc[0] if len(value_counts) > 0 else 0.0
    
    return {
        'uniformity': float(uniformity),
        'top_value_frequency': float(top_value_frequency),
        'unique_count': int(clean_column.nunique())
    }

def analyze_column(column: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a column, including DeepJoin/PLM-compatible contextualization statistics and adaptive sampling.
    Args:
        column: Pandas Series representing a column
    Returns:
        Dictionary with analysis results
    """
    uniqueness_ratio = calculate_uniqueness_ratio(column)
    cardinality = calculate_cardinality(column)
    data_type = detect_data_type(column)
    # Treat booleans as a form of categorical for analytics purposes
    original_data_type = data_type
    if data_type == 'boolean':
        data_type = 'categorical'

    # Handle list types separately by flattening tokens for stats while preserving list nature
    list_delimiter = None  # Track detected delimiter for metadata
    is_list_col = data_type in ['list_numeric', 'list_string']
    if is_list_col:
        # Decide delimiter by first non-null value
        sample_non_null = column.dropna().astype(str).iloc[0] if not column.dropna().empty else ''
        for d in [',', ';', '|', ' ']:
            if d in sample_non_null:
                list_delimiter = d
                break
        if list_delimiter is None:
            list_delimiter = ','
        # Split all cells into tokens
        token_lists = column.dropna().astype(str).apply(lambda x: [t for t in re.split(r'[ ,;|]+', x) if t])
        token_lengths = token_lists.apply(len)
        # Flatten tokens for further numeric / categorical analysis
        flattened_tokens = [tok for sublist in token_lists for tok in sublist]
        flattened_series = pd.Series(flattened_tokens)
        # Recompute mode values based on flattened tokens
        mode_series = flattened_series.mode(dropna=True)
        mode_values = mode_series.tolist()[:5] if not mode_series.empty else []
        # Update data_type for downstream processing
        if data_type == 'list_numeric':
            data_type = 'integer' if all(tok.isdigit() for tok in flattened_tokens) else 'float'
        else:
            data_type = 'string'
        # Replace column reference for stats with flattened values where appropriate
        if not flattened_series.empty:
            # Store extra metadata
            extra_list_stats = {
                'tokens_per_cell_min': int(token_lengths.min()) if not token_lengths.empty else 0,
                'tokens_per_cell_max': int(token_lengths.max()) if not token_lengths.empty else 0,
                'tokens_per_cell_avg': float(token_lengths.mean()) if not token_lengths.empty else 0.0,
                'list_delimiter': list_delimiter
            }
        else:
            extra_list_stats = {
                'tokens_per_cell_min': 0,
                'tokens_per_cell_max': 0,
                'tokens_per_cell_avg': 0.0,
                'list_delimiter': list_delimiter
            }
    else:
        flattened_series = None  # keep linters happy
        extra_list_stats = {}

    # Decide which series to use for downstream stats
    stats_series = flattened_series if is_list_col and flattened_series is not None else column

    missing_percentage = calculate_missing_percentage(column)  # still based on original cells
    # Only compute value length statistics for string-like data
    if data_type == 'string':
        value_lengths = calculate_value_length_stats(column)
    else:
        value_lengths = {}

    distribution = analyze_value_distribution(stats_series)
    total_count = len(stats_series)
    if not is_list_col:  # mode already computed for list columns
        mode_series = stats_series.mode(dropna=True)
        mode_values = mode_series.tolist()[:5] if not mode_series.empty else []
    min_value = max_value = avg_value = ''
    quantiles = {}
    sampled_values = []
    contextual_description = ''
    if uniqueness_ratio < 0.05:
        cardinality_type = 'low'
    elif uniqueness_ratio < 0.5:
        cardinality_type = 'medium'
    else:
        cardinality_type = 'high'

    # Numerical columns
    if data_type in ['integer', 'float']:
        # Coerce to numeric in case column contains integer-like strings
        numeric_col = pd.to_numeric(column, errors='coerce')
        try:
            min_value = float(numeric_col.min()) if not numeric_col.dropna().empty else ''
            max_value = float(numeric_col.max()) if not numeric_col.dropna().empty else ''
            avg_value = float(numeric_col.mean()) if not numeric_col.dropna().empty else ''
            quantiles = numeric_col.quantile([0.25, 0.5, 0.75]).to_dict() if not numeric_col.dropna().empty else {}
        except Exception:
            min_value = max_value = avg_value = ''
            quantiles = {}
        # Distribution descriptor (skew, kurtosis, etc.)
        desc = []
        if not column.dropna().empty:
            try:
                skew = float(column.skew())
                if skew > 1:
                    desc.append('skewed right')
                elif skew < -1:
                    desc.append('skewed left')
                else:
                    desc.append('symmetric')
            except Exception:
                pass
            try:
                kurt = float(column.kurtosis())
                if kurt > 3:
                    desc.append('heavy tails')
                elif kurt < 3:
                    desc.append('light tails')
            except Exception:
                pass
        contextual_description = ', '.join(desc)
        if cardinality > 10000:
            sampled_values = []  # No raw samples
        else:
            sampled_values = mode_values[:5] if cardinality < 100 else mode_values[:3]
    # Temporal columns (date/datetime)
    elif data_type in ['date', 'datetime']:
        try:
            min_value = str(column.min()) if not column.dropna().empty else ''
            max_value = str(column.max()) if not column.dropna().empty else ''
            avg_value = ''
        except Exception:
            min_value = max_value = ''
        contextual_description = f"start: {min_value}, end: {max_value}"
        sampled_values = mode_values[:3]
    # Timedelta/duration columns
    elif data_type == 'timedelta':
        try:
            min_value = str(column.min()) if not column.dropna().empty else ''
            max_value = str(column.max()) if not column.dropna().empty else ''
            avg_value = str(column.mean()) if not column.dropna().empty else ''
        except Exception:
            min_value = max_value = avg_value = ''
        contextual_description = f"duration range: {min_value} to {max_value}"
        sampled_values = []
    # Categorical columns
    elif data_type == 'categorical':
        sampled_values = mode_values[:5] if cardinality < 100 else mode_values[:3]
        contextual_description = f"{cardinality} unique"
    # Text/multi-value columns
    elif data_type == 'string':
        if is_list_col:
            # For list columns use flattened tokens for samples
            sampled_values = flattened_series.sample(min(5, len(flattened_series)), random_state=42).tolist() if flattened_series is not None and not flattened_series.empty else []
            contextual_description = f"list column with delimiter '{list_delimiter}', cardinality {cardinality}"
        elif cardinality > 10000:
            sampled_values = []
            contextual_description = f"{cardinality} unique, text format"
        elif cardinality > 1000:
            sampled_values = mode_values[:3]
            contextual_description = f"{cardinality} unique, text format"
        else:
            # Truncate to 512 tokens or 5 samples
            samples = column.dropna().astype(str).unique().tolist()
            sampled_values = samples[:5]
            contextual_description = f"{cardinality} unique, sample values"
    else:
        sampled_values = mode_values[:3]
        contextual_description = f"{cardinality} unique"

    # Flag for low-cardinality numeric columns that are effectively categorical
    # Use a percentage threshold for 'effectively categorical' numeric columns
    is_effectively_categorical = False
    EFFECTIVE_CATEGORICAL_PERCENT = 0.05  # 5% unique values or fewer is categorical
    # Only check if total_count > 0 to avoid division by zero
    if data_type in ['integer', 'float'] and total_count > 0:
        if (cardinality / total_count) <= EFFECTIVE_CATEGORICAL_PERCENT:
            is_effectively_categorical = True

    # Base analysis structure
    analysis = {
        'column_name': column.name,
        'original_column_name': column.name,
        'uniqueness_ratio': uniqueness_ratio,
        'cardinality': cardinality,
        'data_type': original_data_type if original_data_type == 'boolean' else data_type,
        'missing_percentage': missing_percentage,
        'distribution': distribution,
        'total_count': total_count,
        'mode_values': mode_values,
        'sampled_values': sampled_values,
        'contextual_description': contextual_description,
        'cardinality_type': cardinality_type,
        'is_effectively_categorical': is_effectively_categorical,
    }

    # Include list-specific metadata if present
    if extra_list_stats:
        analysis.update(extra_list_stats)

    # Add optional fields only when meaningful
    if data_type == 'string':
        analysis['value_lengths'] = value_lengths
    if data_type in ['integer', 'float']:
        analysis.update({
            'min_value': min_value,
            'max_value': max_value,
            'avg_value': avg_value,
            'quantiles': quantiles,
        })
    elif data_type in ['date', 'datetime']:
        analysis.update({
            'min_value': min_value,
            'max_value': max_value,
        })
    # categorical/boolean: no numeric stats

    return analysis


from typing import Optional, Dict, Any, List, Tuple


def analyze_all_columns(df: pd.DataFrame, llm_to_original: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all columns in a DataFrame, optionally referencing original column names.
    Args:
        df: Pandas DataFrame
        llm_to_original: Optional dict mapping LLM-inferred column names to original names
    Returns:
        Dictionary mapping LLM-inferred column names to their analysis results (with both names)
    """
    from typing import cast  # local import to avoid polluting module namespace if not needed elsewhere

    results = {}
    for column_name in df.columns:
        original_name = llm_to_original[column_name] if llm_to_original and column_name in llm_to_original else column_name

        # Select the column. pandas returns either a Series or a DataFrame depending on the input
        if original_name in df.columns:
            _col = df[original_name]
        else:
            _col = df[column_name]

        # If a single-column DataFrame is returned, squeeze it down to a Series so that the
        # downstream analytics functions receive the expected type at runtime, and help the
        # static type-checker by casting explicitly.
        if isinstance(_col, pd.DataFrame):
            _col = _col.squeeze(axis=1)

        col_series: pd.Series = cast(pd.Series, _col)

        analysis = analyze_column(col_series)
        analysis['column_name'] = column_name
        analysis['original_column_name'] = original_name
        results[column_name] = analysis
    return results
