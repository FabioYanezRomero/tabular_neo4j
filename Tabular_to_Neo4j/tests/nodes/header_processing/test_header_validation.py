"""
Tests for the header validation node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.header_processing.header_validation import (
    validate_header_llm_node,
)
from Tabular_to_Neo4j.app_state import GraphState


@pytest.mark.unit
def test_validate_header_llm_node_missing_data(runnable_config):
    """Test that the validate_header_llm_node handles missing data."""
    # Create initial state without required data
    initial_state = GraphState.from_dict({"error_messages": []})

    # Call the node
    result_state = validate_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "validated_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert "Cannot validate header" in result_state["error_messages"][0]


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_validation.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_validation.call_llm_with_json_output"
)
def test_validate_header_llm_node_correct_headers(
    mock_call_llm,
    mock_format_prompt,
    sample_dataframe,
    runnable_config,
    mock_llm_response,
):
    """Test that the validate_header_llm_node handles correct headers."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Validate the following headers: {current_header}"
    # Mock the LLM response to indicate headers are correct
    mock_call_llm.return_value = {
        "is_correct": True,
        "validated_header": list(sample_dataframe.columns),
        "suggestions": "",
    }

    # Create initial state with DataFrame and headers
    initial_state = GraphState.from_dict(
        {
            "raw_dataframe": sample_dataframe,
            "final_header": list(sample_dataframe.columns),
            "error_messages": [],
        }
    )

    # Call the node
    result_state = validate_header_llm_node(initial_state, runnable_config)

    # Check that the headers were validated
    assert "validated_header" in result_state
    assert result_state["validated_header"] == list(sample_dataframe.columns)
    assert result_state["final_header"] == list(
        sample_dataframe.columns
    )  # Unchanged since headers are correct
    assert result_state["error_messages"] == []

    # Verify the LLM was called with the correct state name
    mock_call_llm.assert_called_once()
    assert mock_call_llm.call_args[1]["state_name"] == "validate_header"


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_validation.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_validation.call_llm_with_json_output"
)
def test_validate_header_llm_node_improved_headers(
    mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config
):
    """Test that the validate_header_llm_node improves headers when needed."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Validate the following headers: {current_header}"
    # Mock the LLM response to indicate headers need improvement
    mock_call_llm.return_value = {
        "is_correct": False,
        "validated_header": [
            "identifier",
            "full_name",
            "years",
            "email_address",
            "location",
        ],
        "suggestions": "Made headers more descriptive",
    }

    # Create initial state with DataFrame and headers
    initial_state = GraphState.from_dict(
        {
            "raw_dataframe": sample_dataframe,
            "final_header": list(sample_dataframe.columns),
            "error_messages": [],
        }
    )

    # Call the node
    result_state = validate_header_llm_node(initial_state, runnable_config)

    # Check that the headers were improved
    assert "validated_header" in result_state
    assert result_state["validated_header"] == [
        "identifier",
        "full_name",
        "years",
        "email_address",
        "location",
    ]
    assert result_state["final_header"] == [
        "identifier",
        "full_name",
        "years",
        "email_address",
        "location",
    ]
    assert result_state["error_messages"] == []


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_validation.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_validation.call_llm_with_json_output"
)
def test_validate_header_llm_node_invalid_response(
    mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config
):
    """Test that the validate_header_llm_node handles invalid LLM responses."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Validate the following headers: {current_header}"
    # Mock the LLM to return an invalid response
    mock_call_llm.return_value = {
        "is_correct": False,
        "validated_header": "Not a list",  # Should be a list
        "suggestions": "Made headers more descriptive",
    }

    # Create initial state with DataFrame and headers
    initial_state = GraphState.from_dict(
        {
            "raw_dataframe": sample_dataframe,
            "final_header": list(sample_dataframe.columns),
            "error_messages": [],
        }
    )

    # Call the node
    result_state = validate_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert len(result_state["error_messages"]) > 0
    assert "LLM did not return a list of headers" in result_state["error_messages"][0]
