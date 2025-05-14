"""
Tests for the cypher generation node.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Tabular_to_Neo4j.nodes.db_schema.cypher_generation import generate_cypher_templates_node
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
    assert len(result_state['error_messages']) > 0
    assert "Cannot generate Cypher templates" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.db_schema.cypher_generation.format_prompt')
@patch('Tabular_to_Neo4j.nodes.db_schema.cypher_generation.call_llm_with_json_output')
def test_generate_cypher_templates_node_success(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the generate_cypher_templates_node successfully generates Cypher templates."""
    # Mock LLM response
    mock_call_llm.return_value = {
        'load_query': 'LOAD CSV WITH HEADERS FROM "file:///people.csv" AS row CREATE (p:Person {id: row.id, name: row.name})',
        'constraint_queries': ['CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE'],
        'index_queries': ['CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)'],
        'example_queries': ['MATCH (p:Person) RETURN p']
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
            }
        },
        'entity_relationships': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that the Cypher templates were added to the state
    assert 'cypher_query_templates' in result_state
    assert 'load_query' in result_state['cypher_query_templates']
    assert 'constraint_queries' in result_state['cypher_query_templates']
    assert 'index_queries' in result_state['cypher_query_templates']
    assert 'example_queries' in result_state['cypher_query_templates']
    assert 'LOAD CSV' in result_state['cypher_query_templates']['load_query']
    assert len(result_state['cypher_query_templates']['constraint_queries']) == 1
    assert len(result_state['cypher_query_templates']['index_queries']) == 1
    assert len(result_state['cypher_query_templates']['example_queries']) == 1

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.db_schema.cypher_generation.format_prompt')
@patch('Tabular_to_Neo4j.nodes.db_schema.cypher_generation.call_llm_with_json_output')
def test_generate_cypher_templates_node_empty_response(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the generate_cypher_templates_node handles empty LLM responses."""
    # Mock empty LLM response
    mock_call_llm.return_value = {
        'load_query': '',
        'constraint_queries': [],
        'index_queries': [],
        'example_queries': []
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
            }
        },
        'entity_relationships': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that fallback templates were created
    assert 'cypher_query_templates' in result_state
    assert 'load_query' in result_state['cypher_query_templates']
    assert 'LOAD CSV' in result_state['cypher_query_templates']['load_query']

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.db_schema.cypher_generation.format_prompt')
@patch('Tabular_to_Neo4j.nodes.db_schema.cypher_generation.call_llm_with_json_output')
def test_generate_cypher_templates_node_llm_error(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the generate_cypher_templates_node handles LLM errors."""
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
                    {'column_name': 'id', 'property_key': 'id', 'is_identifier': True},
                    {'column_name': 'name', 'property_key': 'name', 'is_identifier': False}
                ]
            }
        },
        'entity_relationships': []
    })
    
    # Call the node
    result_state = generate_cypher_templates_node(initial_state, runnable_config)
    
    # Check that fallback templates were created
    assert 'cypher_query_templates' in result_state
    assert 'load_query' in result_state['cypher_query_templates']
    assert 'Error generating Cypher templates' in result_state['error_messages'][0]
