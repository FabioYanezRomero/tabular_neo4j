"""
Tests for the entity reconciliation node.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Tabular_to_Neo4j.nodes.entity_inference.entity_reconciliation import (
    reconcile_entity_property_node,
)
from Tabular_to_Neo4j.app_state import GraphState


@pytest.mark.unit
def test_reconcile_entity_property_node_missing_data(runnable_config):
    """Test that the reconcile_entity_property_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState.from_dict({"error_messages": []})

    # Call the node
    result_state = reconcile_entity_property_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert len(result_state["error_messages"]) > 0
    assert (
        "Cannot reconcile entity/property classifications"
        in result_state["error_messages"][0]
    )


@pytest.mark.unit
def test_reconcile_entity_property_node_success(runnable_config):
    """Test that the reconcile_entity_property_node successfully reconciles entity and property classifications."""

    # Create initial state with required data
    initial_state = GraphState.from_dict(
        {
            "error_messages": [],
            "csv_file_path": "/path/to/people.csv",
            "entity_property_classification": {
                "id": {
                    "column_name": "id",
                    "classification": "entity_identifier",
                    "entity_type": "Person",
                    "relationship_to_primary": "",
                    "property_name": "id",
                    "reasoning": "This is a unique identifier for a person.",
                    "analytics": {"uniqueness": 1.0},
                    "semantics": {"semantic_type": "identifier"},
                },
                "name": {
                    "column_name": "name",
                    "classification": "entity_property",
                    "entity_type": "Person",
                    "relationship_to_primary": "",
                    "property_name": "name",
                    "reasoning": "This is a property of a person.",
                    "analytics": {"uniqueness": 0.9},
                    "semantics": {"semantic_type": "name"},
                },
            },
        }
    )

    # Call the node
    result_state = reconcile_entity_property_node(initial_state, runnable_config)

    # Check that the consensus was added to the state
    assert "entity_property_consensus" in result_state
    assert len(result_state["entity_property_consensus"]) == 2
    assert "id" in result_state["entity_property_consensus"]
    assert "name" in result_state["entity_property_consensus"]
