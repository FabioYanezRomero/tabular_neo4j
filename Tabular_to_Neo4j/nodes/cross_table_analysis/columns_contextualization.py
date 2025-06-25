"""
Node for cross-table column analytics contextualization.
This node generates contextualized text for each column across all tables for use with encoder language models.
"""
from typing import Dict, Any
from Tabular_to_Neo4j.app_state import MultiTableGraphState

def generate_text_sequence(column_meta: Dict[str, Any], table_title: str = "") -> str:
    """
    Generate contextualized text for a column using DeepJoin/PLM-compatible adaptive patterns.
    Uses cardinality, type, analytics fields, and is_effectively_categorical flag to select the most informative and efficient description.
    """
    column_name = column_meta.get('column_name', '')
    n = column_meta.get('unique_count') or column_meta.get('cardinality') or 0
    min_v = column_meta.get('min_value', '')
    max_v = column_meta.get('max_value', '')
    avg = column_meta.get('avg_value', '')
    quantiles = column_meta.get('quantiles', {})
    mode_values = column_meta.get('mode_values', [])
    sampled_values = column_meta.get('sampled_values', [])
    data_type = column_meta.get('data_type', '')
    desc = column_meta.get('contextual_description', '')
    cardinality_type = column_meta.get('cardinality_type', '')
    is_effectively_categorical = column_meta.get('is_effectively_categorical', False)

    # Treat as categorical if flagged, regardless of original data_type
    if is_effectively_categorical:
        if cardinality_type == 'low':
            values = ', '.join(str(v) for v in sampled_values)
            return f"{column_name} ({n} unique): {values}"
        else:
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({n} unique): {modes}"

    # Numerical columns
    if data_type in ['integer', 'float']:
        quant_str = f", Q1: {quantiles.get(0.25, '')}, Median: {quantiles.get(0.5, '')}, Q3: {quantiles.get(0.75, '')}" if quantiles else ''
        if cardinality_type == 'high':
            return f"{column_name} ({n} unique, min: {min_v}, max: {max_v}, μ: {avg}{quant_str}, {desc})"
        else:
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({n} unique, min: {min_v}, max: {max_v}, μ: {avg}{quant_str}): {modes}"
    # Temporal columns
    if data_type in ['date', 'datetime']:
        return f"{column_name} ({n} unique, start: {min_v}, end: {max_v})"
    # Categorical columns
    if data_type == 'categorical':
        if cardinality_type == 'low':
            values = ', '.join(str(v) for v in sampled_values)
            return f"{column_name} ({n} unique): {values}"
        else:
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({n} unique): {modes}"
    # Text/multi-value columns
    if data_type == 'string':
        if cardinality_type == 'high':
            return f"{column_name} ({n} unique, text format)"
        elif cardinality_type == 'medium':
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({n} unique, text format): {modes}"
        else:
            # Truncate to 512 tokens or 5 samples
            values = ', '.join(str(v) for v in sampled_values)
            return f"{column_name}: {values}"
    # Fallback: use sampled values if available
    if sampled_values:
        values = ', '.join(str(v) for v in sampled_values)
        return f"{column_name}: {values}"
    # As last resort, just column name
    return column_name

from typing import cast

def columns_contextualization_node(state: "MultiTableGraphState", config: Dict[str, Any] = None) -> "MultiTableGraphState":
    """
    Cross-table node that generates contextualized text for each column in all tables.
    Expects state to be a MultiTableGraphState-like dict.
    Returns a new state with a 'columns_contextualization' key containing contextualized text for each column.
    """
    for table_name, table_state in state.items():
        table_title = table_state.get('table_context', table_name)
        analytics = table_state.get('column_analytics', {})
        contextualized = []
        for col_name, col_meta in analytics.items():
            # Compose metadata structure
            column_metadata = {
                "table_name": table_name,
                "column_name": col_name,
                "data_type": col_meta.get("data_type"),
                "unique_count": col_meta.get("cardinality"),
                "total_count": col_meta.get("total_count"),
                "mode_values": col_meta.get("mode_values"),
                "sample_values": col_meta.get("sample_values"),
                "min_value": col_meta.get("min_value"),
                "max_value": col_meta.get("max_value"),
                "null_percentage": col_meta.get("missing_percentage"),
                "table_context": table_title,
            }
            contextualized.append({
                "table": table_name,
                "column": col_name,
                "contextualization": generate_text_sequence(column_metadata, table_title)
            })
        table_state['columns_contextualization'] = contextualized
    return state
