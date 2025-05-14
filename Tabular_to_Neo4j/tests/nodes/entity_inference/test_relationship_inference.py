"""
Tests for the relationship inference node.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Tabular_to_Neo4j.nodes.entity_inference.relationship_inference import infer_entity_relationships_node
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
    assert len(result_state['error_messages']) > 0
    assert "Cannot infer entity relationships" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.relationship_inference.format_prompt')
@patch('Tabular_to_Neo4j.nodes.entity_inference.relationship_inference.call_llm_with_json_output')
def test_infer_entity_relationships_node_success(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the infer_entity_relationships_node successfully infers relationships."""
    # Mock LLM response
    mock_call_llm.return_value = {
        'relationships': [
            {
                'source_entity': 'Person',
                'relationship_type': 'LIVES_IN',
                'target_entity': 'City',
                'properties': [],
                'cardinality': 'MANY_TO_ONE'
            }
        ]
    }
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'property_entity_mapping': {
            'Person': {
                'type': 'entity',
                'entity_type': 'Person',
                'is_primary': True,
                'properties': [
                    {'column_name': 'id', 'property_key': 'id', 'is_identifier': True},
                    {'column_name': 'name', 'property_key': 'name', 'is_identifier': False}
                ]
            },
            'City': {
                'type': 'entity',
                'entity_type': 'City',
                'is_primary': False,
                'properties': [
                    {'column_name': 'city_id', 'property_key': 'cityId', 'is_identifier': True},
                    {'column_name': 'city_name', 'property_key': 'cityName', 'is_identifier': False}
                ]
            }
        }
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that the relationships were added to the state
    assert 'entity_relationships' in result_state
    assert len(result_state['entity_relationships']) == 1
    assert result_state['entity_relationships'][0]['source_entity'] == 'Person'
    assert result_state['entity_relationships'][0]['relationship_type'] == 'LIVES_IN'
    assert result_state['entity_relationships'][0]['target_entity'] == 'City'

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.relationship_inference.call_llm_with_json_output')
def test_infer_entity_relationships_node_llm_error(mock_call_llm, runnable_config):
    """Test that the infer_entity_relationships_node handles LLM errors."""
    # Mock LLM error
    mock_call_llm.side_effect = Exception("LLM error")
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'property_entity_mapping': {
            'Person': {
                'type': 'entity',
                'entity_type': 'Person',
                'is_primary': True,
                'properties': [
                    {'column_name': 'id', 'property_key': 'id', 'is_identifier': True}
                ]
            },
            'City': {
                'type': 'entity',
                'entity_type': 'City',
                'is_primary': False,
                'properties': [
                    {'column_name': 'city_id', 'property_key': 'cityId', 'is_identifier': True}
                ]
            }
        }
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that default relationships were created
    assert 'entity_relationships' in result_state
    assert len(result_state['entity_relationships']) > 0
    assert 'LLM relationship inference failed' in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.relationship_inference.format_prompt')
@patch('Tabular_to_Neo4j.nodes.entity_inference.relationship_inference.call_llm_with_json_output')
def test_infer_entity_relationships_node_no_relationships(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the infer_entity_relationships_node creates default relationships when none are inferred."""
    # Mock LLM response with no relationships
    mock_call_llm.return_value = {
        'relationships': []
    }
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'property_entity_mapping': {
            'Person': {
                'type': 'entity',
                'entity_type': 'Person',
                'is_primary': True,
                'properties': [
                    {'column_name': 'id', 'property_key': 'id', 'is_identifier': True}
                ]
            },
            'City': {
                'type': 'entity',
                'entity_type': 'City',
                'is_primary': False,
                'properties': [
                    {'column_name': 'city_id', 'property_key': 'cityId', 'is_identifier': True}
                ]
            }
        }
    })
    
    # Call the node
    result_state = infer_entity_relationships_node(initial_state, runnable_config)
    
    # Check that default relationships were created
    assert 'entity_relationships' in result_state
    assert len(result_state['entity_relationships']) > 0
    assert result_state['entity_relationships'][0]['source_entity'] == 'Person'
    assert 'HAS_CITY' in result_state['entity_relationships'][0]['relationship_type']
    assert result_state['entity_relationships'][0]['target_entity'] == 'City'
