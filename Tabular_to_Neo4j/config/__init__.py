"""
Configuration package for Tabular_to_Neo4j.
"""

# Default LMStudio model (should match config/lmstudio_config.py logic)
import os
DEFAULT_LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "local-model")

# CSV configuration
CSV_ENCODING = "utf-8"
CSV_DELIMITER = ","
MAX_SAMPLE_ROWS = 10

# Analysis configuration
UNIQUENESS_THRESHOLD = 0.8

from .settings import LLM_CONFIGS, DEFAULT_LMSTUDIO_MODEL, DEFAULT_SEED, DEFAULT_TEMPERATURE