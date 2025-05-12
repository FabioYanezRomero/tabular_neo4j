"""
Tests for the schema finalization node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization import synthesize_final_schema_node
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
    assert 'final_schema' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot synthesize final schema" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization.call_llm_with_json_output')
def test_synthesize_final_schema_node_success(mock_call_llm, mock_format_prompt, runnable_config, mock_llm_response):
    """Test that the synthesize_final_schema_node successfully synthesizes the final schema."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Synthesize the final schema for the given entities and relationships"
    # Mock the LLM response
    mock_call_llm.return_value = mock_llm_response("synthesize_final_schema")
    
    # Create initial state with required data
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
    
    cypher_query_templates = {
        'load_query': "LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row MERGE (p:Person {id: row.id}) SET p.name = row.name, p.age = toInteger(row.age), p.email = row.email",
        'constraint_queries': [
            'CREATE CONSTRAINT ON (p:Person) ASSERT p.id IS UNIQUE'
        ],
        'index_queries': [],
        'example_queries': [
            'MATCH (p:Person) RETURN p LIMIT 10',
            'MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p, c'
        ]
    }
    
    constraints_and_indexes = [
        {
            'type': 'CONSTRAINT',
            'entity_type': 'Person',
            'property': 'id',
            'query': 'CREATE CONSTRAINT ON (p:Person) ASSERT p.id IS UNIQUE'
        }
    ]
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'entity_relationships': entity_relationships,
        'cypher_query_templates': cypher_query_templates,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that the final schema was added to the state
    assert 'inferred_neo4j_schema' in result_state
    assert isinstance(result_state['inferred_neo4j_schema'], dict)
    assert 'node_labels' in result_state['inferred_neo4j_schema']
    assert 'relationship_types' in result_state['inferred_neo4j_schema']
    assert 'property_keys' in result_state['inferred_neo4j_schema']
    assert 'constraints' in result_state['inferred_neo4j_schema']
    assert 'indexes' in result_state['inferred_neo4j_schema']
    assert result_state['error_messages'] == []
    
    # Verify the LLM was called with the correct state name
    assert mock_call_llm.call_count > 0
    assert mock_call_llm.call_args[1]['state_name'] == "synthesize_final_schema"

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization.call_llm_with_json_output')
def test_synthesize_final_schema_node_empty_data(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the synthesize_final_schema_node handles empty data."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Synthesize the final schema for the given entities and relationships"
    # Create initial state with empty data
    initial_state = GraphState({
        'property_entity_mapping': {},
        'entity_relationships': [],
        'csv_file_path': '/path/to/person.csv',
        'error_messages': []
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that the node handled the empty data gracefully
    assert 'inferred_neo4j_schema' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot synthesize final schema: missing required data:" in result_state['error_messages'][0]
    
    # Verify the LLM was not called
    mock_call_llm.assert_not_called()

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization.format_prompt')
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.schema_finalization.call_llm_with_json_output')
def test_synthesize_final_schema_node_llm_error(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the synthesize_final_schema_node handles LLM errors gracefully."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Synthesize the final schema for the given entities and relationships"
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
    
    cypher_query_templates = {
        'load_query': "LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row MERGE (p:Person {id: row.id}) SET p.name = row.name",
        'constraint_queries': [
            'CREATE CONSTRAINT ON (p:Person) ASSERT p.id IS UNIQUE'
        ],
        'index_queries': [],
        'example_queries': [
            'MATCH (p:Person) RETURN p LIMIT 10'
        ]
    }
    
    initial_state = GraphState({
        'property_entity_mapping': property_entity_mapping,
        'entity_relationships': entity_relationships,
        'cypher_query_templates': cypher_query_templates,
        'csv_file_path': '/path/to/person.csv',  # For primary entity inference (will derive 'Person')
        'error_messages': []
    })
    
    # Call the node
    result_state = synthesize_final_schema_node(initial_state, runnable_config)
    
    # Check that the node handled the error gracefully but created a fallback schema
    assert 'inferred_neo4j_schema' in result_state
    assert isinstance(result_state['inferred_neo4j_schema'], dict)
    assert 'node_labels' in result_state['inferred_neo4j_schema']
    assert 'relationship_types' in result_state['inferred_neo4j_schema']
    assert 'property_keys' in result_state['inferred_neo4j_schema']
    assert len(result_state['error_messages']) > 0
    assert "Error synthesizing final schema:" in result_state['error_messages'][0]
