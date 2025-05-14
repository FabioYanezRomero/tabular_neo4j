"""
Tests for the property mapping node.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Tabular_to_Neo4j.nodes.entity_inference.property_mapping import map_properties_to_entities_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_map_properties_to_entities_node_missing_data(runnable_config):
    """Test that the map_properties_to_entities_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = map_properties_to_entities_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert len(result_state['error_messages']) > 0
    assert "Cannot map properties to entities" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.property_mapping.format_prompt')
@patch('Tabular_to_Neo4j.nodes.entity_inference.property_mapping.call_llm_with_json_output')
def test_map_properties_to_entities_node_success(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the map_properties_to_entities_node successfully maps properties to entities."""
    # Mock LLM response
    mock_call_llm.return_value = {
        'properties': [
            {
                'column_name': 'id',
                'property_key': 'id',
                'is_identifier': True
            },
            {
                'column_name': 'name',
                'property_key': 'name',
                'is_identifier': False
            }
        ]
    }
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'entity_property_consensus': {
            'id': {
                'column_name': 'id',
                'classification': 'entity_identifier',
                'entity_type': 'Person',
                'relationship_to_primary': '',
                'neo4j_property_key': 'id',
                'semantic_type': 'identifier',
                'confidence': 0.9,
                'reasoning': 'This is a unique identifier for a person.'
            },
            'name': {
                'column_name': 'name',
                'classification': 'entity_property',
                'entity_type': 'Person',
                'relationship_to_primary': '',
                'neo4j_property_key': 'name',
                'semantic_type': 'name',
                'confidence': 0.8,
                'reasoning': 'This is a property of a person.'
            }
        }
    })
    
    # Call the node
    result_state = map_properties_to_entities_node(initial_state, runnable_config)
    
    # Check that the mapping was added to the state
    assert 'property_entity_mapping' in result_state
    assert 'Person' in result_state['property_entity_mapping']
    assert result_state['property_entity_mapping']['Person']['type'] == 'entity'
    assert len(result_state['property_entity_mapping']['Person']['properties']) == 2

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.property_mapping.call_llm_with_json_output')
def test_map_properties_to_entities_node_llm_error(mock_call_llm, runnable_config):
    """Test that the map_properties_to_entities_node handles LLM errors."""
    # Mock LLM error
    mock_call_llm.side_effect = Exception("LLM error")
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'entity_property_consensus': {
            'id': {
                'column_name': 'id',
                'classification': 'entity_identifier',
                'entity_type': 'Person',
                'relationship_to_primary': '',
                'neo4j_property_key': 'id',
                'semantic_type': 'identifier',
                'confidence': 0.9,
                'reasoning': 'This is a unique identifier for a person.'
            }
        }
    })
    
    # Call the node
    result_state = map_properties_to_entities_node(initial_state, runnable_config)
    
    # Check that the mapping was added to the state with fallback values
    assert 'property_entity_mapping' in result_state
    assert 'Person' in result_state['property_entity_mapping']
    assert result_state['property_entity_mapping']['Person']['type'] == 'entity'
    assert len(result_state['property_entity_mapping']['Person']['properties']) == 1
    assert 'Error mapping properties for entity' in result_state['error_messages'][0]
