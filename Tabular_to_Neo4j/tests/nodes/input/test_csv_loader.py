"""
Tests for the CSV loader node.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState
from Tabular_to_Neo4j.nodes.input.csv_loader import load_csv_node


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.input.csv_loader.load_csv_safely")
def test_load_csv_node_success(mock_load_csv, sample_csv_content, runnable_config):
    """Test that the load_csv_node successfully loads a CSV file."""
    # Mock the load_csv_safely function to return a valid DataFrame
    mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    mock_load_csv.return_value = (mock_df, ["col1", "col2"], "utf-8")

    # Create initial state with the CSV file path
    initial_state = GraphState.from_dict(
        {"csv_file_path": sample_csv_content, "error_messages": []}
    )

    # Call the node
    result_state = load_csv_node(initial_state, runnable_config)

    # Check that the raw_dataframe was added to the state
    assert "raw_dataframe" in result_state
    assert isinstance(result_state["raw_dataframe"], pd.DataFrame)
    assert "potential_header" in result_state
    assert "encoding_used" in result_state
    assert len(result_state["raw_dataframe"]) == 3  # 3 rows in our mock data
    assert list(result_state["raw_dataframe"].columns) == ["col1", "col2"]
    assert result_state["error_messages"] == []


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.input.csv_loader.load_csv_safely")
def test_load_csv_node_file_not_found(mock_load_csv, runnable_config):
    """Test that the load_csv_node handles file not found errors."""
    # Mock the load_csv_safely function to return None and an error
    mock_load_csv.return_value = (
        None,
        ["File not found: /path/to/nonexistent/file.csv"],
        None,
    )

    # Create initial state with a non-existent CSV file path
    initial_state = GraphState.from_dict(
        {"csv_file_path": "/path/to/nonexistent/file.csv", "error_messages": []}
    )

    # Call the node
    result_state = load_csv_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "error_messages" in result_state
    assert len(result_state["error_messages"]) > 0
    assert "raw_dataframe" not in result_state
    assert "Failed to load CSV file" in result_state["error_messages"][0]


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.input.csv_loader.load_csv_safely")
def test_load_csv_node_invalid_csv(mock_load_csv, runnable_config):
    """Test that the load_csv_node handles CSV files with parsing issues."""
    # In real-world scenarios, we often encounter CSV files with parsing issues such as:
    # - Inconsistent number of columns
    # - Quoting issues
    # - Encoding problems
    # This test simulates a CSV file that was loaded but had parsing warnings

    # Create a mock DataFrame that represents what we could extract from a problematic CSV
    mock_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["John Doe", "Jane Smith", "Error in row", "Bob Johnson"],
            "email": ["john@example.com", "jane@example.com", None, "bob@example.com"],
        }
    )

    # Realistic parsing errors that might be encountered
    mock_errors = [
        "Warning: Line 3 has parsing issues - missing fields",
        "Warning: Inconsistent quoting detected in file",
    ]

    # Set the mock return value
    mock_load_csv.return_value = (mock_df, mock_errors, "utf-8")

    # Create initial state with a CSV file path
    # In a real scenario, this would be a path to an actual file
    initial_state = GraphState.from_dict(
        {"csv_file_path": "/path/to/problematic_data.csv", "error_messages": []}
    )

    # Call the node
    result_state = load_csv_node(initial_state, runnable_config)

    # Check that the raw_dataframe was added to the state despite the parsing issues
    assert "raw_dataframe" in result_state
    assert isinstance(result_state["raw_dataframe"], pd.DataFrame)
    assert len(result_state["raw_dataframe"]) == 4  # 4 rows in our mock data

    # Check that the warnings are handled appropriately
    # In the actual implementation, parsing warnings are logged but not added to error_messages
    # Instead, the implementation tracks specific issues in dedicated state fields
    assert "error_messages" in result_state

    # Check that columns with nulls were detected (this is what the implementation actually does)
    assert "columns_with_nulls" in result_state
    assert "email" in result_state["columns_with_nulls"]

    # Check that other expected fields are present
    assert "potential_header" in result_state
    assert "encoding_used" in result_state
    assert result_state["encoding_used"] == "utf-8"
