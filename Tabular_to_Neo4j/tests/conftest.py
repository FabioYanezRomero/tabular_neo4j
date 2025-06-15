"""
Test fixtures for Tabular to Neo4j converter tests.
"""

import os
import pandas as pd
import pytest
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from Tabular_to_Neo4j.app_state import GraphState


@pytest.fixture
def sample_csv_path():
    """Fixture that returns a path to a sample CSV file for testing."""
    # Create a path to a sample CSV in the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(tests_dir, "test_data", "sample.csv")


@pytest.fixture
def ensure_test_data_dir(sample_csv_path):
    """Ensure the test data directory exists."""
    os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
    return os.path.dirname(sample_csv_path)


@pytest.fixture
def sample_csv_content(ensure_test_data_dir, sample_csv_path):
    """Create a sample CSV file for testing."""
    content = """id,name,age,email,city
1,John Doe,30,john@example.com,New York
2,Jane Smith,25,jane@example.com,Los Angeles
3,Bob Johnson,40,bob@example.com,Chicago
4,Alice Brown,35,alice@example.com,Houston
5,Charlie Wilson,28,charlie@example.com,Phoenix
"""
    with open(sample_csv_path, "w") as f:
        f.write(content)
    return sample_csv_path


@pytest.fixture
def sample_dataframe():
    """Fixture that returns a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": [
                "John Doe",
                "Jane Smith",
                "Bob Johnson",
                "Alice Brown",
                "Charlie Wilson",
            ],
            "age": [30, 25, 40, 35, 28],
            "email": [
                "john@example.com",
                "jane@example.com",
                "bob@example.com",
                "alice@example.com",
                "charlie@example.com",
            ],
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        }
    )


@pytest.fixture
def sample_graph_state(sample_dataframe, sample_csv_path):
    """Fixture that returns a sample GraphState for testing."""
    return GraphState.from_dict(
        {
            "csv_file_path": sample_csv_path,
            "raw_dataframe": sample_dataframe.copy(),
            "processed_dataframe": sample_dataframe.copy(),
            "has_header": True,
            "detected_header": ["id", "name", "age", "email", "city"],
            "final_header": ["id", "name", "age", "email", "city"],
            "error_messages": [],
        }
    )


@pytest.fixture
def runnable_config():
    """Fixture that returns a sample RunnableConfig for testing."""
    return RunnableConfig({"configurable": {"thread_id": "test-thread"}})


@pytest.fixture
def mock_llm_response():
    """Fixture that provides mock LLM responses for different state names."""

    def _get_response(state_name: str) -> Dict[str, Any]:
        responses = {
            "infer_header": {
                "inferred_headers": ["id", "name", "age", "email", "city"],
                "confidence": "high",
            },
            "validate_header": {
                "validated_headers": ["id", "name", "age", "email", "city"],
                "suggestions": [],
                "is_valid": True,
            },
            "analyze_column_semantics": {
                "column_semantics": {
                    "id": {
                        "type": "identifier",
                        "description": "Unique identifier for the person",
                    },
                    "name": {
                        "type": "personal_name",
                        "description": "Full name of the person",
                    },
                    "age": {"type": "numeric_age", "description": "Age in years"},
                    "email": {
                        "type": "email_address",
                        "description": "Contact email address",
                    },
                    "city": {"type": "location", "description": "City of residence"},
                }
            },
            "classify_entities_properties": {
                "entities": ["Person"],
                "properties": ["id", "name", "age", "email", "city"],
                "classification": {
                    "id": "property",
                    "name": "property",
                    "age": "property",
                    "email": "property",
                    "city": "property",
                },
            },
            "reconcile_entity_property": {
                "consensus_classification": {
                    "primary_entity": "Person",
                    "columns": {
                        "id": "property",
                        "name": "property",
                        "age": "property",
                        "email": "property",
                        "city": "property",
                    },
                }
            },
            "map_properties_to_entity": {
                "property_entity_mapping": {
                    "Person": ["id", "name", "age", "email", "city"]
                },
                "property_types": {
                    "id": "String",
                    "name": "String",
                    "age": "Integer",
                    "email": "String",
                    "city": "String",
                },
            },
            "infer_entity_relationships": {"entity_relationships": []},
            "generate_cypher_templates": {
                "cypher_templates": [
                    "CREATE (p:Person {id: $id, name: $name, age: $age, email: $email, city: $city})"
                ],
                "constraints_and_indexes": [
                    "CREATE CONSTRAINT person_id_constraint IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
                ],
            },
            "synthesize_final_schema": {
                "final_schema": {
                    "primary_entity_label": "Person",
                    "columns_classification": {
                        "id": {"entity": "Person", "type": "String", "is_key": True},
                        "name": {"entity": "Person", "type": "String", "is_key": False},
                        "age": {"entity": "Person", "type": "Integer", "is_key": False},
                        "email": {
                            "entity": "Person",
                            "type": "String",
                            "is_key": False,
                        },
                        "city": {"entity": "Person", "type": "String", "is_key": False},
                    },
                    "relationships": [],
                    "cypher_templates": [
                        "CREATE (p:Person {id: $id, name: $name, age: $age, email: $email, city: $city})"
                    ],
                    "constraints_and_indexes": [
                        "CREATE CONSTRAINT person_id_constraint IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
                    ],
                }
            },
        }
        return responses.get(state_name, {})

    return _get_response
