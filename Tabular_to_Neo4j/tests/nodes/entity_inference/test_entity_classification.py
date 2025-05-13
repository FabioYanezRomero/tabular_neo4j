"""
Tests for the entity classification node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.entity_inference.entity_classification import classify_entities_properties_node
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
    assert len(result_state['error_messages']) > 0
    assert "Cannot classify entities/properties" in result_state['error_messages'][0]

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.entity_classification.format_prompt')
@patch('Tabular_to_Neo4j.nodes.entity_inference.entity_classification.call_llm_with_json_output')
def test_classify_entities_properties_node_success(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the classify_entities_properties_node successfully classifies entities and properties."""
    # Mock LLM response
    mock_call_llm.return_value = {
        'classification': 'entity',
        'entity_type': 'Person',
        'relationship_to_primary': 'IS_A',
        'property_name': 'person_id',
        'reasoning': 'This is a unique identifier for a person.'
    }
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'final_header': ['id', 'name', 'age'],
        'column_analytics': {
            'id': {'uniqueness': 1.0, 'cardinality': 100, 'data_type': 'integer'},
            'name': {'uniqueness': 0.9, 'cardinality': 90, 'data_type': 'string'},
            'age': {'uniqueness': 0.1, 'cardinality': 10, 'data_type': 'integer'}
        },
        'llm_column_semantics': {
            'id': {'semantic_type': 'identifier', 'neo4j_role': 'ID'},
            'name': {'semantic_type': 'name', 'neo4j_role': 'PROPERTY'},
            'age': {'semantic_type': 'age', 'neo4j_role': 'PROPERTY'}
        },
        'processed_dataframe': None
    })
    
    # Call the node
    result_state = classify_entities_properties_node(initial_state, runnable_config)
    
    # Check that the classification was added to the state
    assert 'entity_property_classification' in result_state
    assert len(result_state['entity_property_classification']) == 3
    assert result_state['entity_property_classification']['id']['classification'] == 'entity'
    assert result_state['entity_property_classification']['id']['entity_type'] == 'Person'

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.entity_inference.entity_classification.format_prompt')
@patch('Tabular_to_Neo4j.nodes.entity_inference.entity_classification.call_llm_with_json_output')
def test_classify_entities_properties_node_llm_error(mock_call_llm, mock_format_prompt, runnable_config):
    """Test that the classify_entities_properties_node handles LLM errors."""
    # Mock LLM error
    mock_call_llm.side_effect = Exception("LLM error")
    
    # Create initial state with required data
    initial_state = GraphState({
        'error_messages': [],
        'csv_file_path': '/path/to/people.csv',
        'final_header': ['id', 'name', 'age'],
        'column_analytics': {
            'id': {'uniqueness': 1.0, 'cardinality': 100, 'data_type': 'integer'},
            'name': {'uniqueness': 0.9, 'cardinality': 90, 'data_type': 'string'},
            'age': {'uniqueness': 0.1, 'cardinality': 10, 'data_type': 'integer'}
        },
        'llm_column_semantics': {
            'id': {'semantic_type': 'identifier', 'neo4j_role': 'ID'},
            'name': {'semantic_type': 'name', 'neo4j_role': 'PROPERTY'},
            'age': {'semantic_type': 'age', 'neo4j_role': 'PROPERTY'}
        },
        'processed_dataframe': None
    })
    
    # Call the node
    result_state = classify_entities_properties_node(initial_state, runnable_config)
    
    # Check that the classification was added to the state with fallback values
    assert 'entity_property_classification' in result_state
    assert len(result_state['entity_property_classification']) == 3
    assert 'LLM entity classification failed' in result_state['error_messages'][0]
    # Check that a fallback classification was used
    assert result_state['entity_property_classification']['id']['classification'] in ['entity', 'property']
