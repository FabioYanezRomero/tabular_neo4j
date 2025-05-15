"""
Configuration package for Tabular_to_Neo4j.
"""

# Default configuration values
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0

# CSV configuration
CSV_ENCODING = "utf-8"
CSV_DELIMITER = ","
MAX_SAMPLE_ROWS = 10

# Analysis configuration
UNIQUENESS_THRESHOLD = 0.8

LLM_CONFIGS = {
    "header_inference": {
        "model": "default",
        "temperature": 0.2
    },
    "relationship_inference": {
        "model": "default",
        "temperature": 0.3
    }
}
