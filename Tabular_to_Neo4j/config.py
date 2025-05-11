"""
Configuration settings for the Tabular to Neo4j converter.
"""

# LLM API Configuration
LLM_API_KEY = ""  # Set your API key here or use environment variables
LLM_MODEL_NAME_GENERAL = "gpt-3.5-turbo"  # For inference, validation, analysis
LLM_MODEL_NAME_TRANSLATE = "gpt-3.5-turbo"  # For translation tasks
TARGET_HEADER_LANGUAGE = "English"  # Target language for headers
LLM_PROVIDER = "openai"  # Options: "openai", "ollama", "anthropic", "lmstudio", "huggingface"

# LMStudio and HuggingFace specific settings
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # Default LMStudio local server
HUGGINGFACE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  # Example model

# CSV Processing Settings
MAX_SAMPLE_ROWS = 10  # Maximum number of rows to include in LLM prompts
CSV_ENCODING = "utf-8"  # Default encoding for CSV files
CSV_DELIMITER = ","  # Default delimiter for CSV files

# Analysis Settings
UNIQUENESS_THRESHOLD = 0.9  # Threshold for considering a column as potentially unique identifier
LOW_CARDINALITY_THRESHOLD = 0.1  # Threshold for considering a column as low cardinality (categorical)
