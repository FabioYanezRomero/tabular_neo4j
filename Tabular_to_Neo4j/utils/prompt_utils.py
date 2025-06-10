"""Utilities for loading and formatting prompts."""

import json
import os
import time
from pathlib import Path
from typing import Dict

from Tabular_to_Neo4j.utils.logging_config import get_logger

logger = get_logger(__name__)

_CURRENT_RUN_TIMESTAMP_DIR = None


def reset_prompt_sample_directory() -> None:
    """Reset the prompt sample directory for a new pipeline run."""
    global _CURRENT_RUN_TIMESTAMP_DIR
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
    template_name: str,
    formatted_prompt: str,
    kwargs: Dict,
    *,
    is_template: bool = False,
    is_error: bool = False,
) -> None:
    """Save a formatted prompt sample to the prompt_samples folder."""
    global _CURRENT_RUN_TIMESTAMP_DIR

    base_dir = Path(__file__).parent.parent.parent
    prompt_samples_dir = base_dir / "prompt_samples"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if _CURRENT_RUN_TIMESTAMP_DIR is None:
        _CURRENT_RUN_TIMESTAMP_DIR = prompt_samples_dir / timestamp
        os.makedirs(_CURRENT_RUN_TIMESTAMP_DIR, exist_ok=True)

    timestamp_dir = _CURRENT_RUN_TIMESTAMP_DIR

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
        "map_properties_to_entities": 11,
        "infer_entity_relationships": 12,
        "generate_cypher_templates": 13,
    }

    state_name = kwargs.get("state_name", "")
    numeric_id = 0
    base_state_name = state_name
    if not base_state_name:
        for node_name in node_order.keys():
            if node_name in template_name:
                base_state_name = node_name
                break
    if base_state_name in node_order:
        numeric_id = node_order[base_state_name]

    file_prefix = "template" if is_template else "error" if is_error else "formatted"
    column_suffix = ""
    if "classify_entities_properties" in template_name and "column_name" in kwargs:
        column_suffix = f"_{kwargs['column_name']}"

    prompt_file = (
        timestamp_dir
        / f"{numeric_id:02d}_{template_name.replace('.txt', '')}{column_suffix}_{file_prefix}.txt"
    )
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(formatted_prompt)

    kwargs_file = (
        timestamp_dir
        / f"{numeric_id:02d}_{template_name.replace('.txt', '')}{column_suffix}_{file_prefix}_kwargs.json"
    )
    with open(kwargs_file, "w", encoding="utf-8") as f:
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(
                value, (str, int, float, bool, list, dict)
            ) and not isinstance(value, type):
                serializable_kwargs[key] = value
            else:
                serializable_kwargs[key] = str(value)
        json.dump(serializable_kwargs, f, indent=2)

    metadata_file = timestamp_dir / "metadata.json"
    if not metadata_file.exists():
        metadata = {
            "run_timestamp": timestamp,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "templates": [],
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        template_info = {
            "template_name": template_name,
            "is_template": is_template,
            "is_error": is_error,
            "file_prefix": file_prefix,
            "node_type": template_name.replace(".txt", ""),
        }
        if "classify_entities_properties" in template_name and "column_name" in kwargs:
            template_info["column_name"] = kwargs["column_name"]
        template_exists = False
        for existing in metadata.get("templates", []):
            if (
                existing.get("template_name") == template_name
                and existing.get("is_template") == is_template
                and existing.get("is_error") == is_error
                and existing.get("column_name", None)
                == template_info.get("column_name", None)
            ):
                template_exists = True
                break
        if not template_exists:
            metadata.setdefault("templates", []).append(template_info)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error updating metadata: {e}")


def format_prompt(template_name: str, **kwargs) -> str:
    """Format a prompt template with the given arguments."""
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

    save_prompt_sample(template_name, template, formatted_kwargs, is_template=True)

    try:
        formatted_prompt = template
        for key, value in formatted_kwargs.items():
            placeholder = "{" + key + "}"
            formatted_prompt = formatted_prompt.replace(placeholder, value)
        save_prompt_sample(template_name, formatted_prompt, formatted_kwargs)
        return formatted_prompt
    except KeyError as e:
        error_msg = f"Error formatting prompt: missing key {e}. Available keys: {list(formatted_kwargs.keys())}"
        logger.error(f"Missing key in prompt template {template_name}: {e}")
        save_prompt_sample(template_name, error_msg, formatted_kwargs, is_error=True)
        return error_msg
    except Exception as e:
        error_msg = f"Error formatting prompt: {str(e)}"
        logger.error(f"Error formatting prompt template {template_name}: {e}")
        save_prompt_sample(template_name, error_msg, formatted_kwargs, is_error=True)
        return error_msg
