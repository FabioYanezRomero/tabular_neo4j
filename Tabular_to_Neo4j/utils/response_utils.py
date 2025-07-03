"""Utility functions for parsing LLM responses."""

import json
import re
from typing import Dict, Any

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_json_from_llm_response(response: str) -> Dict[str, Any]:
    """Extract JSON object or array from a raw LLM response."""
    if not response or response.strip() == "":
        logger.warning("Received empty response from LLM")
        return {"error": "Empty response", "raw_response": ""}

    cleaned_response = response.strip()

    # Remove code block markers
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_response)
    if code_block:
        cleaned_response = code_block.group(1).strip()
    else:
        cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()

    # Remove extra wrapping quotes if present
    if cleaned_response and cleaned_response[0] in "\"'" and cleaned_response[-1] == cleaned_response[0]:
        cleaned_response = cleaned_response[1:-1]

    # Try to match a JSON object or array
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned_response)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if 'Invalid control character' in str(e):
                logger.error("LLM output contains literal newlines or unescaped control characters inside string values. Cannot auto-fix. Returning raw string.")
                return {"error": "Invalid control character in JSON string (likely literal newline in LLM output)", "raw_response": json_str}
            logger.warning(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")

    # Only now try to unescape and parse as a last resort
    import codecs
    try:
        cleaned_unescaped = codecs.decode(cleaned_response, 'unicode_escape')
        return json.loads(cleaned_unescaped)
    except Exception as esc_e:
        logger.warning(f"Failed to unescape and parse: {esc_e}")
        # Last resort: try to extract largest JSON object/array in string
        curly_matches = re.findall(r'\{[\s\S]*?\}', cleaned_response)
        square_matches = re.findall(r'\[[\s\S]*?\]', cleaned_response)
        matches = curly_matches + square_matches
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except Exception:
                continue
        logger.warning("Failed to parse JSON after all attempts.")
        # Fallback: return cleaned string if it looks like JSON
        if cleaned_response and cleaned_response[0] in '{[':
            return {"error": "Failed to parse JSON", "raw_response": cleaned_response}
        return {"error": "Failed to parse JSON", "raw_response": cleaned_response}