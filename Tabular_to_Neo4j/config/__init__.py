"""
Configuration package for Tabular_to_Neo4j.
"""

# Default configuration values
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0

# Default LMStudio model (should match config/lmstudio_config.py logic)
import os
DEFAULT_LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "local-model")

# CSV configuration
CSV_ENCODING = "utf-8"
CSV_DELIMITER = ","
MAX_SAMPLE_ROWS = 10

# Analysis configuration
UNIQUENESS_THRESHOLD = 0.8

LLM_CONFIGS = {
    "header_inference": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for inferring headers when none are detected",
        "auto_load": True,
        "auto_unload": True
    },
    "header_validation": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for validating and improving headers",
        "auto_load": True,
        "auto_unload": True
    },
    "header_translation": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for translating headers to target language",
        "auto_load": True,
        "auto_unload": True
    },
    "entity_classification": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for classifying columns as entities or properties",
        "auto_load": True,
        "auto_unload": True
    },
    "entity_reconciliation": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for reconciling entity/property classifications",
        "auto_load": True,
        "auto_unload": True
    },
    "relationship_inference": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for inferring relationships between entities",
        "auto_load": True,
        "auto_unload": True
    },
    "cypher_generation": {
        "model": "lmstudio-community/gemma-3-12b-it-GGUF",
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for generating Cypher templates",
        "auto_load": True,
        "auto_unload": True
    }
}

