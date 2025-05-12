"""
Tests for the relationship inference node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.schema_synthesis.relationship_inference import infer_entity_relationships_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_infer_entity_relationships_node_missing_data(runnable_config):
    """Test that the infer_entity_relationships_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'entity_relationships' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot infer entity relationships" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.relationship_inference.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.relationship_inference.call_llm_with_json_output')
def test_infer_entity_relationships_node_success(mock_call_llm, mock_format_prompt, runnable_config, mock_llm_response):
    """Test that the infer_entity_relationships_node successfully infers relationships between entities."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Infer relationships between entities for the given data"
    # Mock the LLM response
    mock_call_llm.return_value = mock_llm_response("infer_entity_relationships")
    
    # Create initial state with required data that matches the expected format
    property_entity_mapping = {
        'Person': {
            'type': 'entity',
            'entity_type': 'Person',
            'is_primary': True,
            'properties': [
                {'property_name': 'id', 'property_key': 'id', 'is_identifier': True},
                {'property_name': 'name', 'property_key': 'name', 'is_identifier': False},
                {'property_name': 'age', 'property_key': 'age', 'is_identifier': False},
                {'property_name': 'email', 'property_key': 'email', 'is_identifier': False}
            ]
        },
        'City': {
            'type': 'entity',
            'entity_type': 'City',
            'is_primary': False,
            'properties': [
                {'property_name': 'city', 'property_key': 'name', 'is_identifier': True}
            ]
        }
    }
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'entity_property_consensus': {},  # Add this for completeness
        'error_messages': []
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that the entity relationships were added to the state
    assert 'entity_relationships' in result_state
    assert isinstance(result_state['entity_relationships'], list)
    assert len(result_state['entity_relationships']) > 0
    assert result_state['error_messages'] == []
    
    # Verify the LLM was called with the correct state name
    assert mock_call_llm.call_count > 0
    assert mock_call_llm.call_args[1]['state_name'] == "infer_entity_relationships"

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.relationship_inference.call_llm_with_json_output')
def test_infer_entity_relationships_node_single_entity(mock_call_llm, runnable_config):
    """Test that the infer_entity_relationships_node handles a single entity case."""
    # Create initial state with only one entity
    property_entity_mapping = {
        'Person': {
            'type': 'entity',
            'entity_type': 'Person',
            'is_primary': True,
            'properties': [
                {'property_name': 'id', 'property_key': 'id', 'is_identifier': True},
                {'property_name': 'name', 'property_key': 'name', 'is_identifier': False},
                {'property_name': 'age', 'property_key': 'age', 'is_identifier': False},
                {'property_name': 'email', 'property_key': 'email', 'is_identifier': False}
            ]
        }
    }
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'entity_property_consensus': {},  # Add this for completeness
        'error_messages': []
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that no relationships were inferred (only one entity)
    assert 'entity_relationships' in result_state
    assert isinstance(result_state['entity_relationships'], list)
    assert len(result_state['entity_relationships']) == 0
    assert result_state['error_messages'] == []
    
    # Verify the LLM was not called
    mock_call_llm.assert_not_called()

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.relationship_inference.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.relationship_inference.call_llm_with_json_output')
def test_infer_entity_relationships_node_llm_error(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the infer_entity_relationships_node handles LLM errors gracefully."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Infer relationships between entities for the given data"
    # Mock the LLM to raise an exception
    mock_call_llm.side_effect = Exception("LLM error")
    
    # Create initial state with required data
    property_entity_mapping = {
        'Person': {
            'type': 'entity',
            'entity_type': 'Person',
            'is_primary': True,
            'properties': [
                {'property_name': 'id', 'property_key': 'id', 'is_identifier': True},
                {'property_name': 'name', 'property_key': 'name', 'is_identifier': False}
            ]
        },
        'City': {
            'type': 'entity',
            'entity_type': 'City',
            'is_primary': False,
            'properties': [
                {'property_name': 'city', 'property_key': 'name', 'is_identifier': True}
            ]
        }
    }
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'entity_property_consensus': {},  # Add this for completeness
        'error_messages': []
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that the node handled the error gracefully
    assert 'entity_relationships' in result_state
    assert isinstance(result_state['entity_relationships'], list)
    # The implementation should create default relationships as fallback
    assert len(result_state['entity_relationships']) > 0
    assert len(result_state['error_messages']) > 0
    assert "Error inferring entity relationships:" in result_state['error_messages'][0]
