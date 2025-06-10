"""
Tests for the header inference node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.header_processing.header_inference import (
    infer_header_llm_node,
)
from Tabular_to_Neo4j.app_state import GraphState


@pytest.mark.unit
def test_infer_header_llm_node_no_dataframe(runnable_config):
    """Test that the infer_header_llm_node handles missing DataFrame."""
    # Create initial state without a DataFrame
    initial_state = GraphState.from_dict({"error_messages": []})

    # Call the node
    result_state = infer_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "inferred_header" not in result_state
    assert "final_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert "Cannot infer header" in result_state["error_messages"][0]


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_inference.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_inference.call_llm_with_json_output"
)
def test_infer_header_llm_node_success(
    mock_call_llm,
    mock_format_prompt,
    sample_dataframe,
    runnable_config,
    mock_llm_response,
):
    """Test that the infer_header_llm_node successfully infers headers."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = (
        "Infer headers for the following data: {data_sample}"
    )
    # Mock the LLM response
    mock_call_llm.return_value = list(sample_dataframe.columns)

    # Create initial state with a DataFrame but no headers
    initial_state = GraphState.from_dict(
        {"raw_dataframe": sample_dataframe, "error_messages": []}
    )

    # Call the node
    result_state = infer_header_llm_node(initial_state, runnable_config)

    # Check that the headers were inferred
    assert "inferred_header" in result_state
    assert result_state["inferred_header"] == list(sample_dataframe.columns)
    assert result_state["final_header"] == list(sample_dataframe.columns)
    assert result_state["error_messages"] == []

    # Verify the LLM was called with the correct state name
    mock_call_llm.assert_called_once()
    assert mock_call_llm.call_args[1]["state_name"] == "infer_header"


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_inference.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_inference.call_llm_with_json_output"
)
def test_infer_header_llm_node_invalid_response(
    mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config
):
    """Test that the infer_header_llm_node handles invalid LLM responses."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = (
        "Infer headers for the following data: {data_sample}"
    )
    # Mock the LLM to return an invalid response (not a list)
    mock_call_llm.return_value = {"error": "Not a list of headers"}

    # Create initial state with a DataFrame
    initial_state = GraphState.from_dict(
        {"raw_dataframe": sample_dataframe, "error_messages": []}
    )

    # Call the node
    result_state = infer_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "inferred_header" not in result_state
    assert "final_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert "LLM did not return a list of headers" in result_state["error_messages"][0]


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_inference.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_inference.call_llm_with_json_output"
)
def test_infer_header_llm_node_wrong_length(
    mock_call_llm, mock_format_prompt, sample_dataframe, runnable_config
):
    """Test that the infer_header_llm_node handles LLM responses with wrong number of headers."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = (
        "Infer headers for the following data: {data_sample}"
    )
    # Mock the LLM to return a list with wrong number of headers
    mock_call_llm.return_value = ["header1", "header2"]  # Not enough headers

    # Create initial state with a DataFrame
    initial_state = GraphState.from_dict(
        {"raw_dataframe": sample_dataframe, "error_messages": []}
    )

    # Call the node
    result_state = infer_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "inferred_header" not in result_state
    assert "final_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert "LLM returned 2 headers, but CSV has" in result_state["error_messages"][0]
