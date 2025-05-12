"""
Tests for the entity classification node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.schema_synthesis.entity_classification import classify_entities_properties_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_classify_entities_properties_node_missing_data(runnable_config):
    """Test that the classify_entities_properties_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = classify_entities_properties_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'entity_property_classification' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot classify entities/properties: missing required analysis data" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.entity_classification.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.entity_classification.call_llm_with_json_output')
def test_classify_entities_properties_node_success(mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config):
    """Test that the classify_entities_properties_node correctly classifies entities and properties."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Classify entities and properties for the given data"
    
    # Mock the LLM response with the expected format
    mock_call_llm.return_value = {
        "classification": "property",
        "entity_type": "Person",
        "relationship_to_primary": "",
        "property_name": "id",
        "reasoning": "This is a unique identifier for the Person entity"
    }
    
    # Create initial state with required data using the actual field names from the implementation
    column_analytics = {
        'id': {'data_type': 'numeric', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': [1, 2, 3, 4, 5], 'missing_percentage': 0.0},
        'name': {'data_type': 'string', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'], 'missing_percentage': 0.0},
        'age': {'data_type': 'numeric', 'uniqueness_ratio': 0.8, 'cardinality': 4, 'patterns': [], 'sample_values': [30, 25, 40, 35, 28], 'missing_percentage': 0.0},
        'email': {'data_type': 'string', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': ['@'], 'sample_values': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'], 'missing_percentage': 0.0},
        'city': {'data_type': 'string', 'uniqueness_ratio': 0.8, 'cardinality': 4, 'patterns': [], 'sample_values': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 'missing_percentage': 0.0}
    }
    
    llm_column_semantics = {
        'id': {'semantic_type': 'Identifier', 'neo4j_role': 'PRIMARY_ENTITY_IDENTIFIER', 'description': 'Unique identifier'},
        'name': {'semantic_type': 'PersonName', 'neo4j_role': 'PRIMARY_ENTITY_PROPERTY', 'description': 'Person name'},
        'age': {'semantic_type': 'Age', 'neo4j_role': 'PRIMARY_ENTITY_PROPERTY', 'description': 'Person age'},
        'email': {'semantic_type': 'Email', 'neo4j_role': 'PRIMARY_ENTITY_PROPERTY', 'description': 'Email address'},
        'city': {'semantic_type': 'City', 'neo4j_role': 'RELATED_ENTITY', 'description': 'City name', 'related_entity': 'City'}
    }
    
    initial_state = GraphState({
        'processed_dataframe': sample_dataframe,
        'column_analytics': column_analytics,
        'llm_column_semantics': llm_column_semantics,
        'csv_file_path': '/path/to/customers.csv',  # For primary entity inference
        'final_header': ['id', 'name', 'age', 'email', 'city'],  # Required by the implementation
        'error_messages': []
    })
    
    # Call the node
    result_state = classify_entities_properties_node(initial_state, runnable_config)
    
    # Check that the classifications were added to the state with the correct key name
    assert 'entity_property_classification' in result_state
    assert isinstance(result_state['entity_property_classification'], dict)
    assert len(result_state['entity_property_classification']) > 0
    assert result_state['error_messages'] == []
    
    # In a real-world scenario, we would check that each column has been classified
    for column in ['id', 'name', 'age', 'email', 'city']:
        assert column in result_state['entity_property_classification']
        column_classification = result_state['entity_property_classification'][column]
        assert 'classification' in column_classification  # Should be 'entity' or 'property'
        assert 'entity_type' in column_classification
        assert 'reasoning' in column_classification
    
    # Verify the LLM was called with the correct state name
    assert mock_call_llm.call_count > 0
    assert mock_call_llm.call_args[1]['state_name'] == "classify_entities_properties"

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.entity_classification.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.entity_classification.call_llm_with_json_output')
def test_classify_entities_properties_node_llm_error(mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config):
    """Test that the classify_entities_properties_node handles LLM errors gracefully."""
    # In real-world scenarios, LLM calls can fail for various reasons:
    # - API rate limits
    # - Network issues
    # - Invalid responses
    # This test simulates LLM failures and verifies the fallback behavior
    
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Classify entities and properties for the given data"
    
    # Mock the LLM to raise an exception (simulating an API failure)
    mock_call_llm.side_effect = Exception("LLM API rate limit exceeded")
    
    # Create initial state with required data using the actual field names from the implementation
    column_analytics = {
        'id': {'data_type': 'numeric', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': [1, 2, 3, 4, 5], 'missing_percentage': 0.0},
        'name': {'data_type': 'string', 'uniqueness_ratio': 1.0, 'cardinality': 5, 'patterns': [], 'sample_values': ['John Doe', 'Jane Smith'], 'missing_percentage': 0.0}
    }
    
    llm_column_semantics = {
        'id': {'semantic_type': 'Identifier', 'neo4j_role': 'PRIMARY_ENTITY_IDENTIFIER', 'description': 'Unique identifier', 'related_entity': ''},
        'name': {'semantic_type': 'PersonName', 'neo4j_role': 'PRIMARY_ENTITY_PROPERTY', 'description': 'Person name', 'related_entity': ''}
    }
    
    initial_state = GraphState({
        'processed_dataframe': sample_dataframe[['id', 'name']],  # Just use two columns for simplicity
        'column_analytics': column_analytics,
        'llm_column_semantics': llm_column_semantics,
        'csv_file_path': '/path/to/customers.csv',
        'final_header': ['id', 'name'],  # Required by the implementation
        'error_messages': []
    })
    
    # Call the node
    result_state = classify_entities_properties_node(initial_state, runnable_config)
    
    # In a real-world scenario with LLM failures, the node should provide fallback classifications
    # rather than failing completely, ensuring the pipeline can continue
    assert 'entity_property_classification' in result_state
    assert len(result_state['entity_property_classification']) == 2
    
    # Check that each column has fallback classifications
    for column in ['id', 'name']:
        assert column in result_state['entity_property_classification']
        column_classification = result_state['entity_property_classification'][column]
        
        # Verify the fallback values match what we'd expect in a real scenario
        assert 'classification' in column_classification
        assert 'entity_type' in column_classification
        assert 'reasoning' in column_classification
        assert 'Fallback classification' in column_classification['reasoning']
    
    # Verify that the error was properly logged but didn't stop the process
    assert len(result_state['error_messages']) > 0
    assert mock_call_llm.call_count > 0
    assert "LLM entity classification failed for" in result_state['error_messages'][0]
