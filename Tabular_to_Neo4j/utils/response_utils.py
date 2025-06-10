"""Utility functions for parsing LLM responses."""

import json
import re
from typing import Dict, Any

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_json_from_llm_response(response: str) -> Dict[str, Any]:
    """Extract JSON data from a raw LLM response."""
    if not response or response.strip() == "":
        logger.warning("Received empty response from LLM")
        return {"error": "Empty response", "raw_response": ""}

    logger.debug(
        f"Raw LLM response: {response[:200]}..." if len(response) > 200 else response
    )
    cleaned_response = response.strip()

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_response)
    if json_match:
        json_str = json_match.group(1).strip()
        logger.debug(
            f"Found JSON in code block: {json_str[:100]}..."
            if len(json_str) > 100
            else json_str
        )
    else:
        json_match = re.search(r"(\{[\s\S]*?\})", cleaned_response)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug(
                f"Found JSON with braces: {json_str[:100]}..."
                if len(json_str) > 100
                else json_str
            )
        else:
            json_str = cleaned_response
            logger.debug(
                f"Using whole response as JSON: {json_str[:100]}..."
                if len(json_str) > 100
                else json_str
            )

    if not json_str.startswith("{"):
        brace_index = json_str.find("{")
        if brace_index >= 0:
            json_str = json_str[brace_index:]
            logger.debug(
                f"Trimmed to start at first brace: {json_str[:100]}..."
                if len(json_str) > 100
                else json_str
            )

    if not json_str.endswith("}"):
        brace_index = json_str.rfind("}")
        if brace_index >= 0:
            json_str = json_str[: brace_index + 1]
            logger.debug(
                f"Trimmed to end at last brace: {json_str[:100]}..."
                if len(json_str) > 100
                else json_str
            )

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            potential_jsons = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", json_str)
            if potential_jsons:
                for potential_json in potential_jsons:
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error during JSON extraction: {e}")

    logger.error("Failed to parse JSON from LLM response")
    logger.debug(f"Response was: {response}")
    logger.debug(f"Attempted to parse: {json_str}")

    if "classification" in response.lower() and "column_name" in response.lower():
        if "entity" in response.lower():
            classification = "entity"
        elif "property" in response.lower():
            classification = "property"
        else:
            classification = "unknown"
        return {
            "column_name": (
                re.search(r'column_name[":\s]+(\w+)', response, re.IGNORECASE).group(1)
                if re.search(r'column_name[":\s]+(\w+)', response, re.IGNORECASE)
                else "unknown"
            ),
            "classification": classification,
            "confidence": 0.5,
            "reasoning": "Extracted from unstructured response",
        }
    elif "headers" in response.lower():
        return {"headers": response}
    else:
        return {"error": "Failed to parse JSON", "raw_response": response}
