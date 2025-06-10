"""
Tests for the header translation node.
"""

import pytest
from unittest.mock import patch
from Tabular_to_Neo4j.nodes.header_processing.header_translation import (
    translate_header_llm_node,
)
from Tabular_to_Neo4j.app_state import GraphState


@pytest.mark.unit
def test_translate_header_llm_node_missing_header(runnable_config):
    """Test that the translate_header_llm_node handles missing header."""
    # Create initial state without a header
    initial_state = GraphState.from_dict({"error_messages": []})

    # Call the node
    result_state = translate_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "translated_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert "Cannot translate header" in result_state["error_messages"][0]


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_translation.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_translation.call_llm_with_json_output"
)
def test_translate_header_llm_node_success(
    mock_call_llm, mock_format_prompt, runnable_config, mock_llm_response
):
    """Test that the translate_header_llm_node successfully translates headers."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Translate the following headers from {header_language} to English: {original_header}"
    # Mock the LLM response
    mock_call_llm.return_value = [
        "id",
        "name",
        "age",
        "email",
        "city",
    ]  # Translated headers

    # Create initial state with headers and language info
    initial_state = GraphState.from_dict(
        {
            "final_header": [
                "id",
                "nombre",
                "edad",
                "correo",
                "ciudad",
            ],  # Spanish headers
            "header_language": "Spanish",
            "error_messages": [],
        }
    )

    # Call the node
    result_state = translate_header_llm_node(initial_state, runnable_config)

    # Check that the headers were translated
    assert "translated_header" in result_state
    assert result_state["translated_header"] == ["id", "name", "age", "email", "city"]
    assert result_state["final_header"] == ["id", "name", "age", "email", "city"]
    assert result_state["error_messages"] == []

    # Verify the LLM was called with the correct state name and is_translation flag
    mock_call_llm.assert_called_once()
    assert mock_call_llm.call_args[1]["state_name"] == "translate_header"
    assert mock_call_llm.call_args[1]["is_translation"] is True


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_translation.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_translation.call_llm_with_json_output"
)
def test_translate_header_llm_node_invalid_response(
    mock_call_llm, mock_format_prompt, runnable_config
):
    """Test that the translate_header_llm_node handles invalid LLM responses."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Translate the following headers from {header_language} to English: {original_header}"
    # Mock the LLM to return an invalid response (not a list)
    mock_call_llm.return_value = {"error": "Not a list of headers"}

    # Create initial state with headers
    initial_state = GraphState.from_dict(
        {
            "final_header": ["id", "nombre", "edad", "correo", "ciudad"],
            "header_language": "Spanish",
            "error_messages": [],
        }
    )

    # Call the node
    result_state = translate_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "translated_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert (
        "LLM did not return a list of translated headers"
        in result_state["error_messages"][0]
    )


@pytest.mark.unit
@patch("Tabular_to_Neo4j.nodes.header_processing.header_translation.format_prompt")
@patch(
    "Tabular_to_Neo4j.nodes.header_processing.header_translation.call_llm_with_json_output"
)
def test_translate_header_llm_node_wrong_length(
    mock_call_llm, mock_format_prompt, runnable_config
):
    """Test that the translate_header_llm_node handles LLM responses with wrong number of headers."""
    # Mock the format_prompt to avoid missing template file errors
    mock_format_prompt.return_value = "Translate the following headers from {header_language} to English: {original_header}"
    # Mock the LLM to return a list with wrong number of headers
    mock_call_llm.return_value = ["id", "name", "age"]  # Missing two headers

    # Create initial state with headers
    initial_state = GraphState.from_dict(
        {
            "final_header": ["id", "nombre", "edad", "correo", "ciudad"],
            "header_language": "Spanish",
            "error_messages": [],
        }
    )

    # Call the node
    result_state = translate_header_llm_node(initial_state, runnable_config)

    # Check that an error message was added to the state
    assert "translated_header" not in result_state
    assert len(result_state["error_messages"]) > 0
    assert (
        "LLM returned 3 headers, but original has 5 columns"
        in result_state["error_messages"][0]
    )
