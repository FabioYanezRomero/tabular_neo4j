"""
Tests for the entity reconciliation node.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Tabular_to_Neo4j.nodes.schema_synthesis.entity_reconciliation import reconcile_entity_property_node
from Tabular_to_Neo4j.app_state import GraphState

@pytest.mark.unit
def test_reconcile_entity_property_node_missing_data(runnable_config):
    """Test that the reconcile_entity_property_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState({
        'error_messages': []
    })
    
    # Call the node
    result_state = reconcile_entity_property_node(initial_state, runnable_config)
    
    # Check that an error message was added to the state
    assert 'entity_property_consensus' not in result_state
    assert len(result_state['error_messages']) > 0
    assert "Cannot reconcile entity/property classifications: missing classification data" in result_state['error_messages'][0]

@pytest.mark.unit
def test_reconcile_entity_property_node_success(runnable_config):
    """Test that the reconcile_entity_property_node successfully reconciles entity and property classifications."""
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob']
    })
    
    # Create initial state with required data
    entity_property_classification = {
        'id': {
            'classification': 'property',
            'entity_type': 'Person',
            'reasoning': 'ID is a property of a person'
        }
    }
    
    initial_state = GraphState({
        'entity_property_classification': entity_property_classification,
        'csv_file_path': '/path/to/customers.csv',
        'processed_dataframe': test_df,
        'error_messages': []
    })
    
    # Call the node
    result_state = reconcile_entity_property_node(initial_state, runnable_config)
    
    # Check that the reconciled classifications were added to the state
    assert 'entity_property_consensus' in result_state
    assert isinstance(result_state['entity_property_consensus'], dict)
    assert len(result_state['entity_property_consensus']) > 0
    assert result_state['error_messages'] == []
    
    # Check that the id column was properly reconciled
    assert 'id' in result_state['entity_property_consensus']
    assert result_state['entity_property_consensus']['id']['classification'] == 'property'
    assert result_state['entity_property_consensus']['id']['entity_type'] == 'Person'

@pytest.mark.unit
@patch('Tabular_to_Neo4j.nodes.schema_synthesis.entity_reconciliation.get_primary_entity_from_filename')
def test_reconcile_entity_property_node_with_error(mock_get_primary, runnable_config):
    """Test that the reconcile_entity_property_node handles errors gracefully."""
    # Force an error by making the get_primary_entity_from_filename function raise an exception
    mock_get_primary.side_effect = Exception("Test error in primary entity extraction")
    
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob']
    })
    
    # Create initial state with required data
    entity_property_classification = {
        'id': {
            'classification': 'property',
            'entity_type': 'Person',
            'reasoning': 'ID is a property of a person'
        }
    }
    
    initial_state = GraphState({
        'entity_property_classification': entity_property_classification,
        'csv_file_path': '/path/to/customers.csv',
        'processed_dataframe': test_df,
        'error_messages': []
    })
    
    # Call the node
    result_state = reconcile_entity_property_node(initial_state, runnable_config)
    
    # Check that the error was handled gracefully
    assert 'entity_property_consensus' in result_state
    assert isinstance(result_state['entity_property_consensus'], dict)
    assert len(result_state['error_messages']) > 0
    assert "Error reconciling entity/property classifications:" in result_state['error_messages'][0]
