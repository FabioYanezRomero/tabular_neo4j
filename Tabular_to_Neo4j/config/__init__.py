"""
Configuration package for Tabular_to_Neo4j.
"""

# Default configuration values
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.7
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
