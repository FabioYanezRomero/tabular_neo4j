"""
Tests for the property mapping node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.schema_synthesis.property_mapping import map_properties_to_entities_node as map_properties_to_entity_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_map_properties_to_entity_node_missing_data(runnable_config):
    """Test that the map_properties_to_entity_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = map_properties_to_entity_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'property_entity_mapping' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot map properties to entities" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.property_mapping.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.property_mapping.call_llm_with_json_output')
def test_map_properties_to_entity_node_success(mock_call_llm, mock_format_prompt, runnable_config, mock_llm_response):
    """Test that the map_properties_to_entity_node successfully maps properties to entities."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Map properties to entities for the given data"
    # Mock the LLM response
    mock_call_llm.return_value = mock_llm_response("map_properties_to_entity")
    
    # Create initial state with required data
    entity_property_consensus = {
        'id': {
            'classification': 'entity_property',  # Must match what the implementation expects
            'entity_type': 'Person',
            'neo4j_property_key': 'id'
        },
        'name': {
            'classification': 'entity_property',  # Must match what the implementation expects
            'entity_type': 'Person',
            'neo4j_property_key': 'name'
        },
        'city': {
            'classification': 'new_entity_type',  # Must match what the implementation expects
            'entity_type': 'City',
            'relationship_to_primary': 'LIVES_IN',
            'neo4j_property_key': 'name'
        }
    }
    
    initial_state = GraphState({
        'entity_property_consensus': entity_property_consensus,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = map_properties_to_entity_node(initial_state, runnable_config)
    
    # Check that the property-entity mappings were added to the state
    assert 'property_entity_mapping' in result_state
    assert isinstance(result_state['property_entity_mapping'], dict)
    assert len(result_state['property_entity_mapping']) > 0
    assert result_state['error_messages'] == []
    
    # Verify the LLM was called with the correct state name
    assert mock_call_llm.call_count > 0
    assert mock_call_llm.call_args[1]['state_name'] == "map_properties_to_entity"

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.property_mapping.call_llm_with_json_output')
def test_map_properties_to_entity_node_empty_reconciled_data(mock_call_llm, runnable_config):
    """Test that the map_properties_to_entity_node handles empty reconciled data."""
    # Create initial state with empty reconciled data
    initial_state = GraphState({
        'entity_property_consensus': {},
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = map_properties_to_entity_node(initial_state, runnable_config)
    
    # Check that the node handled the empty data gracefully
    assert 'property_entity_mapping' in result_state
    assert isinstance(result_state['property_entity_mapping'], dict)
    assert len(result_state['property_entity_mapping']) == 0
    assert result_state['error_messages'] == []
    
    # Verify the LLM was not called
    mock_call_llm.assert_not_called()

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.property_mapping.call_llm_with_json_output')
def test_map_properties_to_entity_node_llm_error(mock_call_llm, runnable_config):
    """Test that the map_properties_to_entity_node handles LLM errors gracefully."""
    # Mock the LLM to raise an exception
    mock_call_llm.side_effect = Exception("LLM error")
    
    # Create initial state with required data
    entity_property_consensus = {
        'id': {
            'classification': 'entity_property',  # Must match what the implementation expects
            'entity_type': 'Person',
            'neo4j_property_key': 'id'
        }
    }
    
    initial_state = GraphState({
        'entity_property_consensus': entity_property_consensus,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = map_properties_to_entity_node(initial_state, runnable_config)
    
    # Check that the node handled the error gracefully
    assert 'property_entity_mapping' in result_state
    assert isinstance(result_state['property_entity_mapping'], dict)
    assert len(result_state['error_messages']) > 0
    assert "Error mapping properties to entities" in result_state['error_messages'][0]
