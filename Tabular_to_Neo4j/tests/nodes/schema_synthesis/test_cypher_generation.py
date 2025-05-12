"""
Tests for the Cypher generation node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation import generate_cypher_templates_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_generate_cypher_templates_node_missing_data(runnable_config):
    """Test that the generate_cypher_templates_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'cypher_query_templates' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot generate Cypher templates" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation.call_llm_with_json_output')
def test_generate_cypher_templates_node_success(mock_call_llm, mock_format_prompt, runnable_config, mock_llm_response):
    """Test that the generate_cypher_templates_node successfully generates Cypher templates."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Generate Cypher templates for the given entities and relationships"
    # Mock the LLM response
    mock_call_llm.return_value = mock_llm_response("generate_cypher_templates")
    
    # Create initial state with required data that matches the expected format
    property_entity_mapping = {
        'Person': {
            'type': 'entity',
            'entity_type': 'Person',
            'is_primary': True,
            'properties': [
                {'property_name': 'id', 'property_key': 'id', 'is_identifier': True, 'column_name': 'id'},
                {'property_name': 'name', 'property_key': 'name', 'is_identifier': False, 'column_name': 'name'},
                {'property_name': 'age', 'property_key': 'age', 'is_identifier': False, 'column_name': 'age'},
                {'property_name': 'email', 'property_key': 'email', 'is_identifier': False, 'column_name': 'email'}
            ]
        },
        'City': {
            'type': 'entity',
            'entity_type': 'City',
            'is_primary': False,
            'properties': [
                {'property_name': 'city', 'property_key': 'name', 'is_identifier': True, 'column_name': 'city'}
            ]
        }
    }
    
    entity_relationships = [
        {
            'source_entity': 'Person',
            'target_entity': 'City',
            'relationship_type': 'LIVES_IN',
            'cardinality': 'MANY_TO_ONE',
            'source_column': None,
            'target_column': 'city',
            'properties': []
        }
    ]
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'entity_relationships': entity_relationships,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that the Cypher templates were added to the state
    assert 'cypher_query_templates' in result_state
    assert isinstance(result_state['cypher_query_templates'], dict)
    assert 'load_query' in result_state['cypher_query_templates']
    assert 'constraint_queries' in result_state['cypher_query_templates']
    assert 'index_queries' in result_state['cypher_query_templates']
    assert 'example_queries' in result_state['cypher_query_templates']
    assert result_state['error_messages'] == []
    
    # Verify the LLM was called with the correct state name
    assert mock_call_llm.call_count > 0
    assert mock_call_llm.call_args[1]['state_name'] == "generate_cypher_templates"

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation.call_llm_with_json_output')
def test_generate_cypher_templates_node_empty_entities(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the generate_cypher_templates_node handles empty entity data."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Generate Cypher templates for the given entities and relationships"
    # Create initial state with empty entity data but with a minimal valid structure
    # to avoid errors in the implementation
    property_entity_mapping = {}
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'entity_relationships': [],
        'csv_file_path': '/path/to/person.csv',
        'error_messages': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that the node handled the empty data gracefully
    assert 'cypher_query_templates' in result_state
    assert isinstance(result_state['cypher_query_templates'], dict)
    # We don't check the exact content of the templates since they might be generated
    # differently based on the implementation's fallback mechanism
    assert 'load_query' in result_state['cypher_query_templates']
    assert 'constraint_queries' in result_state['cypher_query_templates']
    assert 'index_queries' in result_state['cypher_query_templates']
    assert 'example_queries' in result_state['cypher_query_templates']
    # The implementation doesn't add an error message when there are no entities
    assert result_state['error_messages'] == []
    
    # Verify the LLM was called
    assert mock_call_llm.call_count > 0
    assert mock_call_llm.call_args[1]['state_name'] == "generate_cypher_templates"

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.cypher_generation.call_llm_with_json_output')
def test_generate_cypher_templates_node_llm_error(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the generate_cypher_templates_node handles LLM errors gracefully."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Generate Cypher templates for the given entities and relationships"
    # Mock the LLM to raise an exception
    mock_call_llm.side_effect = Exception("LLM error")
    
    # Create initial state with required data
    property_entity_mapping = {
        'Person': {
            'type': 'entity',
            'entity_type': 'Person',
            'is_primary': True,
            'properties': [
                {'property_name': 'id', 'property_key': 'id', 'is_identifier': True, 'column_name': 'id'},
                {'property_name': 'name', 'property_key': 'name', 'is_identifier': False, 'column_name': 'name'}
            ]
        }
    }
    
    entity_relationships = []
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'entity_relationships': entity_relationships,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that the node handled the error gracefully
    assert 'cypher_query_templates' in result_state
    assert isinstance(result_state['cypher_query_templates'], dict)
    assert 'load_query' in result_state['cypher_query_templates']
    assert 'constraint_queries' in result_state['cypher_query_templates']
    assert 'index_queries' in result_state['cypher_query_templates']
    assert 'example_queries' in result_state['cypher_query_templates']
    assert len(result_state['error_messages']) > 0
    assert "Error generating Cypher templates:" in result_state['error_messages'][0]
