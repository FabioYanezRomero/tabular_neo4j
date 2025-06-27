"""Utilities for loading and formatting prompts."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

_CURRENT_RUN_TIMESTAMP_DIR = None


def reset_prompt_sample_directory(base_dir: str = "samples", timestamp: str = None) -> None:
    """Reset the prompt sample directory for a new pipeline run, optionally with a base_dir and timestamp."""
    global _CURRENT_RUN_TIMESTAMP_DIR
    if timestamp:
        _CURRENT_RUN_TIMESTAMP_DIR = Path(base_dir) / timestamp
    else:
        _CURRENT_RUN_TIMESTAMP_DIR = None


def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    base_dir = Path(__file__).parent.parent
    prompt_path = base_dir / "prompts" / template_name
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error loading prompt template {template_name}: {e}")
        return "Please analyze the following data: {data}"


def save_prompt_sample(
    # TODO: When using multiple-table set table-name to "inter_table" whenever we finish with the individual tables
    template_name: str,
    formatted_prompt: str,
    kwargs: Dict,
    *,
    is_template: bool = False,
    is_error: bool = False,
    base_dir: str = "samples",
    table_name: str,
    unique_suffix: str = "",
) -> None:
    """Save a formatted prompt sample to the output folder, using the global output saver's timestamp."""
    global _CURRENT_RUN_TIMESTAMP_DIR
    
    from Tabular_to_Neo4j.utils.output_saver import output_saver
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized. All prompt saving must use the same timestamp for the run.")
    timestamp = output_saver.timestamp
    if not timestamp:
        raise RuntimeError("No timestamp available from OutputSaver for prompt saving.")
    # Always save inside <base_dir>/<timestamp>/<table_name>/prompts or <base_dir>/<timestamp>/inter_table/prompts
    if not table_name:
        raise ValueError("table_name must be provided for prompt saving")
    out_dir = Path(base_dir) / timestamp / table_name / "prompts"
    os.makedirs(out_dir, exist_ok=True)
    timestamp_dir = out_dir

    node_order = {
        "load_csv": 1,
        "detect_header": 2,
        "infer_header": 3,
        "validate_header": 4,
        "detect_header_language": 5,
        "translate_header": 6,
        "apply_header": 7,
        "analyze_columns": 8,
        "classify_entities_properties": 9,
        "reconcile_entity_property": 10,
        "map_properties_to_entity": 11,
        "entity_relationship_pair": 12,
}

    state_name = kwargs.get("state_name", "")
    base_state_name = state_name
    if not base_state_name:
        for node_name in node_order.keys():
            if node_name in template_name:
                base_state_name = node_name
                break
    numeric_id = node_order.get(base_state_name, 0)

    file_prefix = "template" if is_template else "error" if is_error else "formatted"
    column_suffix = ""
    if "classify_entities_properties" in template_name and "column_name" in kwargs:
        column_suffix = f"_{kwargs['column_name']}"

    suffix = f"_{unique_suffix}" if unique_suffix else ""
    prompt_file = (
        timestamp_dir
        / f"{numeric_id:02d}_{template_name.replace('.txt', '')}{column_suffix}{suffix}_{file_prefix}.txt"
    )
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(formatted_prompt)
    # Do NOT save any kwargs .json or metadata.json files in the prompts folder.
    # Only the .txt file is created.


def format_prompt(template_name: str, *, table_name: Optional[str] = None, unique_suffix: str = "", **kwargs) -> str:
    """
    Format a prompt template with the given arguments.
    Pass table_name or subfolder for correct prompt saving location.
    """
    
    template = load_prompt_template(template_name)

    formatted_kwargs: Dict[str, str] = {}
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            formatted_kwargs[key] = str(value)
        elif isinstance(value, list):
            formatted_kwargs[key] = str(value)
        elif isinstance(value, dict):
            formatted_kwargs[key] = str(value)
        else:
            formatted_kwargs[key] = str(value) if value is not None else ""

    logger.debug(f"[format_prompt] Template loaded for '{template_name}': {template[:100]!r}")
    logger.debug(f"[format_prompt] Variables passed: {kwargs}")

    from Tabular_to_Neo4j.utils.output_saver import output_saver
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized. All prompt saving must use the same timestamp for the run.")
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized. All prompt formatting/saving must use the same timestamp for the run.")
    base_dir = output_saver.base_dir
    timestamp = output_saver.timestamp
    if not timestamp:
        raise RuntimeError("No timestamp available from OutputSaver for prompt formatting/saving.")

    try:
        formatted_prompt = template
        for key, value in formatted_kwargs.items():
            placeholder = "{" + key + "}"
            formatted_prompt = formatted_prompt.replace(placeholder, value)
        # Strict enforcement: check for unsubstituted placeholders
        import re
        unsubstituted = re.findall(r"{[\w_]+}", formatted_prompt)
        if unsubstituted:
            logger.debug(f"[format_prompt] Template: {template_name}\nVariables passed: {formatted_kwargs}\nUnsubstituted: {unsubstituted}\nResulting prompt (first 200 chars): {formatted_prompt[:200]!r}")
            logger.warning(f"Unsubstituted placeholders in prompt template {template_name}: {unsubstituted}. Not saving prompt.")
            return f"Error: unsubstituted placeholders in prompt: {unsubstituted}"
        logger.debug(f"[format_prompt] Final formatted prompt: {formatted_prompt}")
        save_prompt_sample(
            template_name,
            formatted_prompt,
            formatted_kwargs,
            base_dir=base_dir,
            table_name=table_name,
            unique_suffix=unique_suffix,
        )
        return formatted_prompt

    except KeyError as e:
        error_msg = f"Error formatting prompt: missing key {e}. Available keys: {list(formatted_kwargs.keys())}"
        logger.error(f"Missing key in prompt template {template_name}: {e}")
        # Do not save error or template prompts in the prompts folder
        return error_msg
    except Exception as e:
        error_msg = f"Error formatting prompt: {str(e)}"
        logger.error(f"Error formatting prompt template {template_name}: {e}")
        # Do not save error or template prompts in the prompts folder
        return error_msg

