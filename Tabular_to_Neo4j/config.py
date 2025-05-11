"""
Configuration settings for the Tabular to Neo4j converter.
"""

# General LLM Configuration
LLM_API_KEY = ""  # Set your API key here or use environment variables
TARGET_HEADER_LANGUAGE = "English"  # Target language for headers
DEFAULT_LLM_PROVIDER = "lmstudio"  # Using LMStudio for GGUF models
DEFAULT_SEED = 42  # Default seed for reproducibility
DEFAULT_TEMPERATURE = 0.0  # Default temperature (0.0 for deterministic results)

# LMStudio settings for GGUF models
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # Default LMStudio local server

# Per-State LLM Configuration with GGUF models through LMStudio
# Each state has its own model and output format specification
LLM_CONFIGS = {
    # Header processing states
    "infer_header": {
        "provider": "lmstudio",
        "model_name": "Mistral-7B-Instruct-v0.2-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for inferring headers when none are detected",
        "output_format": {
            "type": "json_array",
            "description": "Array of inferred header names, should have the same length as the number of columns in the CSV file",
            "example": "[\"col1\", \"col2\", \"col3\", ...]"
        },
        "auto_load": True,  # Whether to automatically load this model when the state starts
        "auto_unload": True  # Whether to automatically unload this model when the state ends
    },
    "validate_header": {
        "provider": "lmstudio",
        "model_name": "Llama-2-7B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for validating and improving headers",
        "output_format": {
            "type": "json_object",
            "description": "Object with validation results",
            "example": "{\"is_correct\": false, \"validated_header\": [\"id\", \"full_name\", \"age\", \"email\"], \"suggestions\": \"Changed 'name' to 'full_name' for clarity\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "translate_header": {
        "provider": "lmstudio",
        "model_name": "Mistral-7B-Instruct-v0.2-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for translating headers to target language",
        "output_format": {
            "type": "json_object",
            "description": "Object with translation results",
            "example": "{\"is_in_target_language\": false, \"translated_header\": [\"id\", \"name\", \"age\", \"email\"]}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    # Analysis states
    "semantic_analysis": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for semantic analysis of columns",
        "output_format": {
            "type": "json_object",
            "description": "Object with semantic analysis of a column",
            "example": "{\"semantic_type\": \"Email\", \"neo4j_role\": \"PRIMARY_ENTITY_PROPERTY\", \"new_node_label_suggestion\": \"\", \"relationship_type_suggestion\": \"\", \"reasoning\": \"This column contains email addresses which are typically used as properties of entities.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    }
}

# Fallback settings (if LMStudio is not available)
#OPENAI_MODEL_NAME = "gpt-3.5-turbo"

# CSV Processing Settings
MAX_SAMPLE_ROWS = 10  # Maximum number of rows to include in LLM prompts
CSV_ENCODING = "utf-8"  # Default encoding for CSV files
CSV_DELIMITER = ","  # Default delimiter for CSV files

# Analysis Settings
UNIQUENESS_THRESHOLD = 0.9  # Threshold for considering a column as potentially unique identifier
LOW_CARDINALITY_THRESHOLD = 0.1  # Threshold for considering a column as low cardinality (categorical)
