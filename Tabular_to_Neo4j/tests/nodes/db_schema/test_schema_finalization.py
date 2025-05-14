"""
Tests for the schema finalization node.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Tabular_to_Neo4j.nodes.db_schema.schema_finalization import synthesize_final_schema_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_synthesize_final_schema_node_missing_data(runnable_config):
    """Test that the synthesize_final_schema_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert len(result_state['error_messages']) > 0
    assert "Cannot synthesize final schema" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.db_schema.schema_finalization.format_prompt')
@patch('Tabular_to_Neo4j.nodes.db_schema.schema_finalization.call_llm_with_json_output')
def test_synthesize_final_schema_node_success(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the synthesize_final_schema_node successfully synthesizes the final schema."""
    # Mock LLM response
    mock_call_llm.return_value = {
        'node_labels': [
            {
                'label': 'Person',
                'description': 'Represents a person',
                'is_primary': True
            }
        ],
        'relationship_types': [
            {
                'type': 'LIVES_IN',
                'source_label': 'Person',
                'target_label': 'City',
                'description': 'Indicates where a person lives',
                'cardinality': 'MANY_TO_ONE'
            }
        ],
        'property_keys': [
            {
                'key': 'id',
                'data_type': 'STRING',
                'description': 'Unique identifier for a person',
                'belongs_to': 'Person',
                'is_identifier': True
            },
            {
                'key': 'name',
                'data_type': 'STRING',
                'description': 'Name of a person',
                'belongs_to': 'Person',
                'is_identifier': False
            }
        ],
        'constraints': [
            {
                'type': 'UNIQUENESS',
                'entity_type': 'Person',
                'property_key': 'id',
                'description': 'Ensures person id is unique'
            }
        ],
        'indexes': [
            {
                'type': 'BTREE',
                'entity_type': 'Person',
                'property_key': 'name',
                'description': 'Index for faster name lookups'
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
            }
        },
        'entity_relationships': [
            {
                'source_entity': 'Person',
                'relationship_type': 'LIVES_IN',
                'target_entity': 'City',
                'properties': [],
                'cardinality': 'MANY_TO_ONE'
            }
        ],
        'cypher_query_templates': {
            'load_query': 'LOAD CSV WITH HEADERS FROM "file:///people.csv" AS row CREATE (p:Person {id: row.id, name: row.name})',
            'constraint_queries': ['CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE'],
            'index_queries': ['CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name)'],
            'example_queries': ['MATCH (p:Person) RETURN p']
        }
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that the final schema was added to the state
    assert 'inferred_neo4j_schema' in result_state
    assert 'node_labels' in result_state['inferred_neo4j_schema']
    assert 'relationship_types' in result_state['inferred_neo4j_schema']
    assert 'property_keys' in result_state['inferred_neo4j_schema']
    assert 'constraints' in result_state['inferred_neo4j_schema']
    assert 'indexes' in result_state['inferred_neo4j_schema']
    assert 'cypher_templates' in result_state['inferred_neo4j_schema']
    assert len(result_state['inferred_neo4j_schema']['node_labels']) == 1
    assert len(result_state['inferred_neo4j_schema']['relationship_types']) == 1
    assert len(result_state['inferred_neo4j_schema']['property_keys']) == 2
    assert len(result_state['inferred_neo4j_schema']['constraints']) == 1
    assert len(result_state['inferred_neo4j_schema']['indexes']) == 1

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.db_schema.schema_finalization.format_prompt')
@patch('Tabular_to_Neo4j.nodes.db_schema.schema_finalization.call_llm_with_json_output')
def test_synthesize_final_schema_node_empty_response(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the synthesize_final_schema_node handles empty LLM responses."""
    # Mock empty LLM response
    mock_call_llm.return_value = {
        'node_labels': [],
        'relationship_types': [],
        'property_keys': [],
        'constraints': [],
        'indexes': []
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
        'entity_relationships': [],
        'cypher_query_templates': {
            'load_query': 'LOAD CSV WITH HEADERS FROM "file:///people.csv" AS row CREATE (p:Person {id: row.id, name: row.name})',
            'constraint_queries': [],
            'index_queries': [],
            'example_queries': []
        }
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that fallback schema was created
    assert 'inferred_neo4j_schema' in result_state
    assert len(result_state['inferred_neo4j_schema']['node_labels']) > 0
    assert len(result_state['inferred_neo4j_schema']['property_keys']) > 0

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.db_schema.schema_finalization.format_prompt')
@patch('Tabular_to_Neo4j.nodes.db_schema.schema_finalization.call_llm_with_json_output')
def test_synthesize_final_schema_node_llm_error(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the synthesize_final_schema_node handles LLM errors."""
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
            }
        },
        'entity_relationships': [],
        'cypher_query_templates': {
            'load_query': 'LOAD CSV WITH HEADERS FROM "file:///people.csv" AS row CREATE (p:Person {id: row.id})',
            'constraint_queries': [],
            'index_queries': [],
            'example_queries': []
        }
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that fallback schema was created
    assert 'inferred_neo4j_schema' in result_state
    assert 'node_labels' in result_state['inferred_neo4j_schema']
    assert 'property_keys' in result_state['inferred_neo4j_schema']
    assert 'Error synthesizing final schema' in result_state['error_messages'][0]
