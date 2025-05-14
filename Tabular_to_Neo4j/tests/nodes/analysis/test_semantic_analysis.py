"""
Tests for the semantic analysis node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.analysis.semantic_analysis import llm_semantic_column_analysis_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_llm_semantic_column_analysis_node_missing_data(runnable_config):
    """Test that the llm_semantic_column_analysis_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = llm_semantic_column_analysis_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'llm_column_semantics' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot perform semantic analysis: missing required data" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.analysis.semantic_analysis.format_prompt')
@patch('Tabular_to_Neo4j.nodes.analysis.semantic_analysis.call_llm_with_json_output')
def test_llm_semantic_column_analysis_node_success(mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config, mock_llm_response):
    """Test that the llm_semantic_column_analysis_node successfully analyzes column semantics."""
    # In real-world scenarios, we need to mock both the prompt formatting and LLM response
    # This simulates a successful LLM analysis despite potential missing prompt templates
    
    # Mock the format_prompt to avoid the missing template file error
    mock_format_prompt.return_value = "Analyze column semantics for the given data"
    
    # Mock the LLM response with realistic column semantics
    mock_llm_response_data = {
        'semantic_type': 'Identifier',
        'neo4j_role': 'PRIMARY_KEY',
        'description': 'Unique identifier for the entity',
        'related_entity': ''
    }
    mock_call_llm.return_value = mock_llm_response_data
    
    # Create initial state with DataFrame and column analytics
    # Using the actual field names from the implementation
    column_analytics = {
        'id': {'data_type': 'numeric', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': [1, 2, 3, 4, 5], 'missing_percentage': 0.0},
        'name': {'data_type': 'string', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'], 'missing_percentage': 0.0},
        'age': {'data_type': 'numeric', 'uniqueness_ratio': 0.8, 'cardinality': 4, 'patterns': [], 'sample_values': [30, 25, 40, 35, 28], 'missing_percentage': 0.0},
        'email': {'data_type': 'string', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': ['@'], 'sample_values': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'], 'missing_percentage': 0.0},
        'city': {'data_type': 'string', 'uniqueness_ratio': 0.8, 'cardinality': 4, 'patterns': [], 'sample_values': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 'missing_percentage': 0.0}
    }
    
    initial_state = GraphState({
        'processed_dataframe': sample_dataframe,
        'column_analytics': column_analytics,
        'csv_file_path': '/path/to/customers.csv',  # For primary entity inference
        'error_messages': []
    })
    
    # Call the node
    result_state = llm_semantic_column_analysis_node(initial_state, runnable_config)
    
    # Check that the semantic analysis results were added to the state
    assert 'llm_column_semantics' in result_state
    assert isinstance(result_state['llm_column_semantics'], dict)
    assert len(result_state['llm_column_semantics']) == len(sample_dataframe.columns)
    
    # In a real-world scenario, we'd check that the semantic analysis has the expected structure
    # rather than specific values which would depend on the LLM response
    for column in sample_dataframe.columns:
        assert column in result_state['llm_column_semantics']
        column_semantics = result_state['llm_column_semantics'][column]
        assert 'semantic_type' in column_semantics
        assert 'neo4j_role' in column_semantics
        assert 'description' in column_semantics
        assert 'related_entity' in column_semantics
    
    # Verify the LLM was called the expected number of times (once per column)
    assert mock_call_llm.call_count == len(sample_dataframe.columns)

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.analysis.semantic_analysis.format_prompt')
@patch('Tabular_to_Neo4j.nodes.analysis.semantic_analysis.call_llm_with_json_output')
def test_llm_semantic_column_analysis_node_llm_error(mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config):
    """Test that the llm_semantic_column_analysis_node handles LLM errors gracefully."""
    # In real-world scenarios, LLM calls can fail for various reasons:
    # - API rate limits
    # - Network issues
    # - Invalid responses
    # This test simulates LLM failures and verifies the fallback behavior
    
    # Mock the format_prompt to avoid the missing template file error
    mock_format_prompt.return_value = "Analyze column semantics for the given data"
    
    # Mock the LLM to raise an exception (simulating an API failure)
    mock_call_llm.side_effect = Exception("LLM API rate limit exceeded")
    
    # Create initial state with DataFrame and column analytics using the actual field names
    column_analytics = {
        'id': {'data_type': 'numeric', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': [1, 2, 3, 4, 5], 'missing_percentage': 0.0},
        'name': {'data_type': 'string', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': ['John Doe', 'Jane Smith'], 'missing_percentage': 0.0}
    }
    
    initial_state = GraphState({
        'processed_dataframe': sample_dataframe[['id', 'name']],  # Just use two columns for simplicity
        'column_analytics': column_analytics,
        'csv_file_path': '/path/to/customers.csv',
        'error_messages': []
    })
    
    # Call the node
    result_state = llm_semantic_column_analysis_node(initial_state, runnable_config)
    
    # In a real-world scenario with LLM failures, the node should provide fallback classifications
    # rather than failing completely, ensuring the pipeline can continue
    assert 'llm_column_semantics' in result_state
    assert len(result_state['llm_column_semantics']) == 2
    
    # Check that each column has fallback semantics with all expected fields
    for column in ['id', 'name']:
        assert column in result_state['llm_column_semantics']
        column_semantics = result_state['llm_column_semantics'][column]
        
        # Verify the fallback values match what we'd expect in a real scenario
        assert column_semantics['semantic_type'] == 'Unknown'
        assert column_semantics['neo4j_role'] == 'PROPERTY'
        assert 'Error in LLM analysis' in column_semantics['description']
        assert 'related_entity' in column_semantics
        
    # Verify that the error was properly logged but didn't stop the process
    assert mock_call_llm.call_count > 0
