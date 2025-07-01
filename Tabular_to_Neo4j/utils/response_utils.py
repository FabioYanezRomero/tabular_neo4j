"""Utility functions for parsing LLM responses."""

import json
import re
from typing import Dict, Any

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_json_from_llm_response(response: str):
    """Extract JSON object or array from a raw LLM response."""
    import json, re
    if not response or response.strip() == "":
        logger.warning("Received empty response from LLM")
        return {"error": "Empty response", "raw_response": ""}

    logger.debug(
        f"Raw LLM response: {response[:200]}..." if len(response) > 200 else response
    )
    cleaned_response = response.strip()

    # Try code block (```json ... ``` or ``` ... ```)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_response)
    if json_match:
        json_str = json_match.group(1).strip()
        logger.debug(
            f"Found JSON in code block: {json_str[:100]}..."
            if len(json_str) > 100
            else json_str
        )
    else:
        # Try to match a JSON object
        json_match = re.search(r"(\{[\s\S]*?\})", cleaned_response)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug(
                f"Found JSON with braces: {json_str[:100]}..."
                if len(json_str) > 100
                else json_str
            )
        else:
            # Try to match a JSON array
            json_match = re.search(r"(\[[\s\S]*?\])", cleaned_response)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.debug(
                    f"Found JSON array: {json_str[:100]}..."
                    if len(json_str) > 100
                    else json_str
                )
            else:
                # Fallback: use the whole response
                json_str = cleaned_response
                logger.debug(
                    f"Using whole response as JSON: {json_str[:100]}..."
                    if len(json_str) > 100
                    else json_str
                )

    # Try parsing
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e}. Attempting secondary cleaning.")
        # Second attempt: strip code block markers and whitespace, then try again
        cleaned = json_str.replace('```json', '').replace('```', '').strip()
        # Additional cleaning: unescape double backslashes and newlines
        import codecs
        try:
            # Unescape common escape sequences (\n, \\ etc.)
            cleaned_unescaped = codecs.decode(cleaned, 'unicode_escape')
        except Exception as esc_e:
            logger.warning(f"Failed to unescape string: {esc_e}")
            cleaned_unescaped = cleaned
        # Try parsing if cleaned string starts with { or [
        if cleaned_unescaped and cleaned_unescaped[0] in '{[':
            try:
                return json.loads(cleaned_unescaped)
            except Exception as e2:
                logger.warning(f"Failed to parse JSON after unescaped direct attempt: {e2}")
        # Try to extract all possible JSON objects/arrays, largest first (no recursive regex)
        # This will match the largest {...} or [...] blocks, but not nested
        curly_matches = re.findall(r'\{[\s\S]*?\}', cleaned_unescaped)
        square_matches = re.findall(r'\[[\s\S]*?\]', cleaned_unescaped)
        matches = curly_matches + square_matches
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except Exception as e3:
                continue
        logger.warning("Failed to parse JSON after all attempts.")
        # Fallback: return cleaned string if it looks like JSON
        if cleaned_unescaped and cleaned_unescaped[0] in '{[':
            return {"error": "Failed to parse JSON", "raw_response": cleaned_unescaped}
        return {"error": "Failed to parse JSON", "raw_response": cleaned_response}