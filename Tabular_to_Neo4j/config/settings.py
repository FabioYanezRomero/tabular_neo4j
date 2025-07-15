"""Configuration settings for the Tabular to Neo4j converter."""

import os
from dotenv import load_dotenv, find_dotenv
from .utils import DynamicLLMConfigs, MODEL_NAME_MAPPING

# Load environment variables from a .env file if present
load_dotenv(find_dotenv())


# General LLM Configuration
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")  # [lm_studio, ollama]
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # Not used with LM Studio but kept for compatibility


# LMStudio Configuration
DEFAULT_LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "gemma-3-12b-it")


# Ollama Configuration
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "gemma-3-12b-it")


# Server URLs
if DEFAULT_LLM_PROVIDER == "lm_studio":
    LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
else:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# LLM randomness Settings
DEFAULT_SEED = 42  # Default seed for reproducibility
DEFAULT_TEMPERATURE = 0.0  # Default temperature (0.0 for deterministic results)


# CSV Processing Settings
MAX_SAMPLE_ROWS = 5  # Maximum number of rows to include in LLM prompts
CSV_ENCODING = "utf-8"  # Default encoding for CSV files
CSV_DELIMITER = ","  # Default delimiter for CSV files


# Analysis Settings
UNIQUENESS_THRESHOLD = 0.9  # Threshold for considering a column as potentially unique identifier
LOW_CARDINALITY_THRESHOLD = 0.1  # Threshold for considering a column as low cardinality (categorical)


# NOTE: For Ollama, use MODEL_NAME_MAPPING[model_name] to get the quantized equivalent.
LLM_CONFIGS = DynamicLLMConfigs({
    # Header processing states
    "infer_header": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
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
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
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
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
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
    # Entity detection state
    "detect_table_entities": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING[DEFAULT_OLLAMA_MODEL] if DEFAULT_LLM_PROVIDER == "ollama" else DEFAULT_OLLAMA_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for detecting whether a table contains entity records and listing entity labels",
        "output_format": {
            "type": "json_object",
            "description": "Entity detection result",
            "example": "{\"has_entities\": true, \"entities\": [\"Customer\"], \"reasoning\": \"Contains customer identifiers and descriptive columns\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },

    # Analysis states
    "analyze_column_semantics": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
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
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
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
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
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
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for mapping properties to entities",
        "output_format": {
            "type": "json_object",
            "description": "Object with property-entity mapping",
            "example": "{\"property_entity_mapping\": {\"city\": \"City\", \"address\": \"Customer\"}, \"entity_properties\": {\"City\": [{\"property_name\": \"city\", \"neo4j_property_key\": \"name\", \"is_identifier\": true}]}, \"reasoning\": \"Mapped properties to their most appropriate entities based on semantics.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "infer_entity_relationship_pair": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for inferring relationships between entities",
        "output_format": {
            "type": "json_object",
            "description": "Object with entity relationships",
            "example": "{\"entity_relationships\": [{\"source_entity\": \"Customer\", \"target_entity\": \"City\", \"relationship_type\": \"LIVES_IN\", \"cardinality\": \"MANY_TO_ONE\", \"source_column\": null, \"target_column\": \"city\", \"properties\": []}], \"reasoning\": \"Customers have a many-to-one relationship with cities.\"}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    # "generate_cypher_templates": {
    #     "provider": DEFAULT_LLM_PROVIDER,
    #     "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
    #     "temperature": DEFAULT_TEMPERATURE,
    #     "seed": DEFAULT_SEED,
    #     "description": "Model for generating Cypher query templates",
    #     "output_format": {
    #         "type": "json_object",
    #         "description": "Object with Cypher templates",
    #         "example": "{\"cypher_templates\": [{\"purpose\": \"Create Customer nodes\", \"query\": \"LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row MERGE (c:Customer {id: row.id}) SET c.name = row.name\", \"description\": \"Creates Customer nodes with their properties\"}], \"constraints_and_indexes\": [{\"type\": \"CONSTRAINT\", \"entity_type\": \"Customer\", \"property\": \"id\", \"query\": \"CREATE CONSTRAINT ON (c:Customer) ASSERT c.id IS UNIQUE\"}]}"
    #     },
    #     "auto_load": True,
    #     "auto_unload": True
    # },
    "synthesize_final_schema": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for synthesizing the final Neo4j schema",
        "output_format": {
            "type": "json_object",
            "description": "Object with the final Neo4j schema",
            "example": "{\"primary_entity_label\": \"Customer\", \"columns_classification\": [{\"original_column_name\": \"id\", \"role\": \"PRIMARY_ENTITY_IDENTIFIER\", \"neo4j_property_key\": \"id\", \"semantic_type\": \"Identifier\"}], \"cypher_templates\": [{\"purpose\": \"Create Customer nodes\", \"query\": \"LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row MERGE (c:Customer {id: row.id}) SET c.name = row.name\"}]}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "semantic_embedding_node": {
    "provider": DEFAULT_LLM_PROVIDER,
    "model_name": MODEL_NAME_MAPPING["bge-base-en-v1.5"] if DEFAULT_LLM_PROVIDER == "ollama" else "bge-base-en-v1.5",
    "temperature": DEFAULT_TEMPERATURE,
    "seed": DEFAULT_SEED,
    "description": "Model for inferring cross-table column relations",
    "output_format": {
        "type": "json_object",
        "description": "Object with cross-table column relationship inference",
        "example": "{\"has_relationship\": true, \"relationship_type\": \"foreign_key\", ...}"
    },
    "auto_load": True,
    "auto_unload": True
},
    "llm_relation_node": {
    "provider": DEFAULT_LLM_PROVIDER,
    "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
    "temperature": DEFAULT_TEMPERATURE,
    "seed": DEFAULT_SEED,
    "description": "Model for inferring cross-table column relations",
    "output_format": {
        "type": "json_object",
        "description": "Object with cross-table column relationship inference",
        "example": "{\"has_relationship\": true, \"relationship_type\": \"foreign_key\", ...}"
    },
    "auto_load": True,
    "auto_unload": True
},
    "analyse": {
    "provider": DEFAULT_LLM_PROVIDER,
    "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
    "temperature": DEFAULT_TEMPERATURE,
    "seed": DEFAULT_SEED,
    "description": "Model for semantic analysis of columns",
    "output_format": {
        "type": "json_object",
        "description": "Object with semantic analysis of a column",
        "example": "{\"semantic_type\": \"Email\", \"neo4j_role\": \"PRIMARY_ENTITY_PROPERTY\", \"new_node_label_suggestion\": \"\", \"relationship_type_suggestion\": \"\", \"reasoning\": \"This column contains email addresses which are typically used as properties of entities.\"}"
    },
    "auto_load": True,
    "auto_unload": True
},
    "merge_synonym_entities": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for detecting synonym entity labels across tables",
        "output_format": {
            "type": "json_object",
            "description": "Object with merges list",
            "example": "{\"merges\": []}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "merge_entities_analytics": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for validating entity merges using column analytics",
        "output_format": {
            "type": "json_object",
            "description": "Object with validated_merges list",
            "example": "{\"validated_merges\": []}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "merge_relation_types": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for consolidating relationship types after entity merges",
        "output_format": {
            "type": "json_object",
            "description": "Object with merges list for relation types",
            "example": "{\"merges\": []}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "infer_intra_table_relations": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for inferring relationships among entities within the same table",
        "output_format": {
            "type": "json_object",
            "description": "Object with inferred intra-table relationships among detected entities",
            "example": "{\"entity_relationships\": []}"
        },
        "auto_load": True,
        "auto_unload": True
    },
    "map_column_to_graph_element": {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_name": MODEL_NAME_MAPPING["gemma-3-12b-it"] if DEFAULT_LLM_PROVIDER == "ollama" else "gemma-3-12b-it",
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "description": "Model for mapping columns to graph elements",
        "output_format": {
            "type": "json_object",
            "description": "Object with column to graph element mapping",
            "example": "{\"column_graph_mapping\": {}}"
        },
        "auto_load": True,
        "auto_unload": True
    }
})