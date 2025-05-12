"""
Tests for the column analytics node.
"""

import pytest
import pandas as pd
from Tabular_to_Neo4j.nodes.analysis.column_analytics import perform_column_analytics_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_perform_column_analytics_node_missing_dataframe(runnable_config):
    """Test that the perform_column_analytics_node handles missing DataFrame."""
    # Create initial state without a DataFrame
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = perform_column_analytics_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'column_analytics' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot analyze columns" in result_state['error_messages'][0]

@pytest.mark.unit
def test_perform_column_analytics_node_success(sample_dataframe, runnable_config):
    """Test that the perform_column_analytics_node successfully analyzes columns."""
    # Create initial state with a DataFrame
    initial_state = GraphState({
        'processed_dataframe': sample_dataframe,
        'error_messages': []
    })
    
    # Call the node
    result_state = perform_column_analytics_node(initial_state, runnable_config)
    
    # Check that the column analytics were added to the state
    assert 'column_analytics' in result_state
    assert isinstance(result_state['column_analytics'], dict)
    assert len(result_state['column_analytics']) == len(sample_dataframe.columns)
    assert result_state['error_messages'] == []
    
    # Check that each column has the expected analytics based on the actual implementation
    for column in sample_dataframe.columns:
        assert column in result_state['column_analytics']
        column_stats = result_state['column_analytics'][column]
        assert 'data_type' in column_stats
        assert 'uniqueness_ratio' in column_stats  # The actual field name in the implementation
        assert 'cardinality' in column_stats
        assert 'patterns' in column_stats
        assert 'sample_values' in column_stats
        assert 'missing_percentage' in column_stats

@pytest.mark.unit
def test_perform_column_analytics_node_numeric_column(runnable_config):
    """Test that the perform_column_analytics_node correctly analyzes numeric columns."""
    # Create a DataFrame with a numeric column
    df = pd.DataFrame({
        'numeric_column': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    })
    
    # Create initial state
    initial_state = GraphState({
        'processed_dataframe': df,
        'error_messages': []
    })
    
    # Call the node
    result_state = perform_column_analytics_node(initial_state, runnable_config)
    
    # Check the analytics for the numeric column based on the actual implementation
    column_stats = result_state['column_analytics']['numeric_column']
    assert column_stats['data_type'] == 'numeric'
    assert column_stats['uniqueness_ratio'] == 0.5  # 5 unique values out of 10
    assert column_stats['cardinality'] == 5  # 5 unique values
    assert 'patterns' in column_stats
    assert 'sample_values' in column_stats
    assert 'missing_percentage' in column_stats

@pytest.mark.unit
def test_perform_column_analytics_node_text_column(runnable_config):
    """Test that the perform_column_analytics_node correctly analyzes text columns."""
    # Create a DataFrame with a text column
    df = pd.DataFrame({
        'text_column': ['apple', 'banana', 'apple', 'cherry', 'banana']
    })
    
    # Create initial state
    initial_state = GraphState({
        'processed_dataframe': df,
        'error_messages': []
    })
    
    # Call the node
    result_state = perform_column_analytics_node(initial_state, runnable_config)
    
    # Check the analytics for the text column based on the actual implementation
    column_stats = result_state['column_analytics']['text_column']
    assert column_stats['data_type'] == 'string'  # The actual data type used in the implementation
    assert column_stats['uniqueness_ratio'] == 0.6  # 3 unique values out of 5
    assert column_stats['cardinality'] == 3  # 3 unique values
    assert 'patterns' in column_stats
    assert 'sample_values' in column_stats
    assert 'missing_percentage' in column_stats
