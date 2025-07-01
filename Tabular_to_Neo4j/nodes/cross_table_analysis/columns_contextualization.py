"""
Node for cross-table column analytics contextualization.
This node generates contextualized text for each column across all tables for use with encoder language models.
"""
from typing import Dict, Any
from Tabular_to_Neo4j.app_state import MultiTableGraphState
from Tabular_to_Neo4j.app_state import GraphState
def generate_text_sequence(column_meta: Dict[str, Any]) -> str:
    """
    Generate contextualized text for a column using DeepJoin/PLM-compatible adaptive patterns.
    Uses cardinality, type, analytics fields, uniqueness ratio, and is_effectively_categorical flag to select the most informative and efficient description.
    """
    column_name = column_meta.get('column_name', '')
    unique_count = column_meta.get('unique_count', '')
    min_v = column_meta.get('min_value', '')
    max_v = column_meta.get('max_value', '')
    avg = column_meta.get('avg_value', '')
    quantiles = column_meta.get('quantiles', {})
    mode_values = column_meta.get('mode_values', [])
    sampled_values = column_meta.get('sampled_values', [])
    data_type = column_meta.get('data_type', '')
    desc = column_meta.get('contextual_description', '')
    cardinality_type = column_meta.get('cardinality_type', '')
    uniqueness_ratio = column_meta.get('uniqueness_ratio', None)
    is_effectively_categorical = column_meta.get('is_effectively_categorical', False)

    # Treat as categorical if flagged, regardless of original data_type
    if is_effectively_categorical:
        if cardinality_type == 'low':
            values = ', '.join(str(v) for v in sampled_values)
            ratio_str = f" ({uniqueness_ratio:.1%} unique)" if uniqueness_ratio is not None else ""
            return f"{column_name} ({unique_count} uniques{ratio_str}): {values}"
        else:
            modes = ', '.join(str(m) for m in mode_values)
            ratio_str = f" ({uniqueness_ratio:.1%} unique)" if uniqueness_ratio is not None else ""
            return f"{column_name} ({unique_count} uniques{ratio_str}): {modes}"

    # Numerical columns
    if data_type in ['integer', 'float']:
        quant_str = f", Q1: {quantiles.get(0.25, '')}, Median: {quantiles.get(0.5, '')}, Q3: {quantiles.get(0.75, '')}" if quantiles else ''
        ratio_str = f" ({uniqueness_ratio:.1%} unique)" if uniqueness_ratio is not None else ""
        if cardinality_type == 'high':
            return f"{column_name} ({unique_count} uniques{ratio_str}, min: {min_v}, max: {max_v}, μ: {avg}{quant_str}, {desc})"
        else:
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({unique_count} uniques{ratio_str}, min: {min_v}, max: {max_v}, μ: {avg}{quant_str}): {modes}"
    # Temporal columns
    if data_type in ['date', 'datetime']:
        ratio_str = f" ({uniqueness_ratio:.1%} unique)" if uniqueness_ratio is not None else ""
        return f"{column_name} ({unique_count} uniques{ratio_str}, start: {min_v}, end: {max_v})"
    
    # Categorical columns
    if data_type == 'categorical':
        ratio_str = f" ({uniqueness_ratio:.1%} unique)" if uniqueness_ratio is not None else ""
        if cardinality_type == 'low':
            values = ', '.join(str(v) for v in sampled_values)
            return f"{column_name} ({unique_count} uniques{ratio_str}): {values}"
        else:
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({unique_count} unique{ratio_str}): {modes}"
    
    # Text/multi-value columns
    if data_type == 'string':
        ratio_str = f" ({uniqueness_ratio:.1%} unique)" if uniqueness_ratio is not None else ""
        if cardinality_type == 'high':
            return f"{column_name} ({unique_count} unique{ratio_str} text format, samples: {mode_values})"
        elif cardinality_type == 'medium':
            modes = ', '.join(str(m) for m in mode_values)
            return f"{column_name} ({unique_count} unique{ratio_str} text format, samples: {mode_values})"
        else:
            # If sampled_values is empty, use mode_values instead
            values = ', '.join(str(v) for v in (sampled_values if sampled_values else mode_values))
            return f"{column_name}{ratio_str}: {values}"
    # Fallback: use sampled values if available
    if sampled_values:
        values = ', '.join(str(v) for v in sampled_values)
        return f"{column_name}: {values}"
    # As last resort, just column name
    return column_name

def columns_contextualization_node(state: "MultiTableGraphState", node_order: int) -> "MultiTableGraphState":
    """
    Cross-table node that generates contextualized text for each column in all tables.
    Expects state to be a MultiTableGraphState-like dict.
    Returns a new state with a 'columns_contextualization' key containing contextualized text for each column.
    
    Args:
        state: The current graph state
        node_order: The order of the node in the pipeline
    
    Returns:
        Updated graph state with columns_contextualization
    """
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info('[columns_contextualization_node][BEFORE] Table states: ' + str({k: type(v).__name__ for k,v in state.items()}))

    for table_name, table_state in state.items():
        if not isinstance(table_state, GraphState):
            # Recover: wrap dict in GraphState
            table_state = GraphState(**table_state)
            state[table_name] = table_state
        table_title = table_state.get('table_context', table_name)
        analytics = table_state.get('column_analytics', {})
        contextualized = []
        for col_name, col_meta in analytics.items():
            # Use original column name for analytics lookup, fallback to col_name if not present
            original_col_name = col_meta.get("original_column_name", col_name)
            analytics_meta = analytics.get(original_col_name, col_meta)
            # Compose metadata structure
            column_metadata = {
                "table_name": table_name,
                "column_name": col_name,
                "original_column_name": original_col_name,
                "data_type": analytics_meta.get("data_type"),
                "unique_count": analytics_meta.get("cardinality"),
                "total_count": analytics_meta.get("total_count"),
                "mode_values": analytics_meta.get("mode_values"),
                "sample_values": analytics_meta.get("sample_values"),
                "min_value": analytics_meta.get("min_value"),
                "max_value": analytics_meta.get("max_value"),
                "null_percentage": analytics_meta.get("missing_percentage"),
                "cardinality_type": analytics_meta.get("cardinality_type"),
                "uniqueness_ratio": analytics_meta.get("uniqueness_ratio", 0.0),
            }
            contextualized.append({
                "table": table_name,
                "column": col_name,
                "contextualization": generate_text_sequence(column_metadata)
            })
        table_state['columns_contextualization'] = contextualized
    # Ensure every table state is a GraphState before returning
    for table_name, table_state in state.items():
        assert isinstance(table_state, GraphState), f"columns_contextualization_node: State for '{table_name}' is not a GraphState, got {type(table_state)}"
    logger.info('[columns_contextualization_node][AFTER] Table states: ' + str({k: type(v).__name__ for k,v in state.items()}))
    return state
