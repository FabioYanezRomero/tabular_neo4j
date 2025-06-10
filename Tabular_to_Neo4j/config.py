"""Configuration settings for the Tabular to Neo4j converter."""

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file if present
load_dotenv(find_dotenv())

# General LLM Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # Not used with LM Studio but kept for compatibility
TARGET_HEADER_LANGUAGE = "English"  # Target language for headers
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio")
DEFAULT_SEED = 42  # Default seed for reproducibility
DEFAULT_TEMPERATURE = 0.0  # Default temperature (0.0 for deterministic results)

# LMStudio settings for GGUF models
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

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
        "skip_json_instruction": True,  # Skip adding the standard JSON formatting instruction
        "auto_load": True,
        "auto_unload": True
    },
    # Analysis states
    "analyze_column_semantics": {
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
    },
    # Schema synthesis states
    "classify_entities_properties": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for classifying columns as entities or properties",
        "output_format": {
            "type": "json_object",
            "description": "Object with entity/property classification",
            "example": "{\"column_name\": \"city\", \"classification\": \"new_entity_type\", \"entity_type\": \"City\", \"relationship_to_primary\": \"LOCATED_IN\", \"neo4j_property_key\": \"name\", \"reasoning\": \"This column contains city names which should be modeled as separate nodes.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "reconcile_entity_property": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for reconciling entity/property classifications",
        "output_format": {
            "type": "json_object",
            "description": "Object with consensus classification",
            "example": "{\"column_name\": \"city\", \"consensus_classification\": \"new_entity_type\", \"entity_type\": \"City\", \"relationship_to_primary\": \"LOCATED_IN\", \"neo4j_property_key\": \"name\", \"confidence\": 0.95, \"reasoning\": \"Both analytics and semantic analysis agree this should be a separate entity.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "map_properties_to_entity": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for mapping properties to entities",
        "output_format": {
            "type": "json_object",
            "description": "Object with property-entity mapping",
            "example": "{\"property_entity_mapping\": {\"city\": \"City\", \"address\": \"Customer\"}, \"entity_properties\": {\"City\": [{\"property_name\": \"city\", \"neo4j_property_key\": \"name\", \"is_identifier\": true}]}, \"reasoning\": \"Mapped properties to their most appropriate entities based on semantics.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "infer_entity_relationships": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for inferring relationships between entities",
        "output_format": {
            "type": "json_object",
            "description": "Object with entity relationships",
            "example": "{\"entity_relationships\": [{\"source_entity\": \"Customer\", \"target_entity\": \"City\", \"relationship_type\": \"LIVES_IN\", \"cardinality\": \"MANY_TO_ONE\", \"source_column\": null, \"target_column\": \"city\", \"properties\": []}], \"reasoning\": \"Customers have a many-to-one relationship with cities.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "generate_cypher_templates": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for generating Cypher query templates",
        "output_format": {
            "type": "json_object",
            "description": "Object with Cypher templates",
            "example": "{\"cypher_templates\": [{\"purpose\": \"Create Customer nodes\", \"query\": \"LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row MERGE (c:Customer {id: row.id}) SET c.name = row.name\", \"description\": \"Creates Customer nodes with their properties\"}], \"constraints_and_indexes\": [{\"type\": \"CONSTRAINT\", \"entity_type\": \"Customer\", \"property\": \"id\", \"query\": \"CREATE CONSTRAINT ON (c:Customer) ASSERT c.id IS UNIQUE\"}]}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "synthesize_final_schema": {
        "provider": "lmstudio",
        "model_name": "Llama-2-13B-Chat-GGUF",  # Name of the model in LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for synthesizing the final Neo4j schema",
        "output_format": {
            "type": "json_object",
            "description": "Object with the final Neo4j schema",
            "example": "{\"primary_entity_label\": \"Customer\", \"columns_classification\": [{\"original_column_name\": \"id\", \"role\": \"PRIMARY_ENTITY_IDENTIFIER\", \"neo4j_property_key\": \"id\", \"semantic_type\": \"Identifier\"}], \"cypher_templates\": [{\"purpose\": \"Create Customer nodes\", \"query\": \"LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row MERGE (c:Customer {id: row.id}) SET c.name = row.name\"}]}"
        },
        "auto_load": True,
        "auto_unload": True
    }
}

# Default model to use if not specified in LLM_CONFIGS
DEFAULT_LMSTUDIO_MODEL = "Mistral-7B-Instruct-v0.2-GGUF"  # Default model for LM Studio

# CSV Processing Settings
MAX_SAMPLE_ROWS = 5  # Maximum number of rows to include in LLM prompts
CSV_ENCODING = "utf-8"  # Default encoding for CSV files
CSV_DELIMITER = ","  # Default delimiter for CSV files

# Analysis Settings
UNIQUENESS_THRESHOLD = 0.9  # Threshold for considering a column as potentially unique identifier
LOW_CARDINALITY_THRESHOLD = 0.1  # Threshold for considering a column as low cardinality (categorical)
