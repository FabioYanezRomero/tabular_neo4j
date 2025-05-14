# Tabular to Neo4j Converter Tests

This directory contains tests for the Tabular to Neo4j converter. The tests are organized by component and use pytest as the testing framework.

## Test Structure

The tests are organized into the following directories:

- `nodes/input/`: Tests for the input nodes (CSV loading, header detection)
- `nodes/header_processing/`: Tests for the header processing nodes (inference, validation, translation)
- `nodes/analysis/`: Tests for the analysis nodes (column analytics, semantic analysis)
- `nodes/schema_synthesis/`: Tests for the schema synthesis nodes (entity classification, reconciliation, etc.)
- `test_main.py`: Integration tests for the entire graph flow

## Running Tests

### Using pytest directly

To run all tests:

```bash
cd /app/Tabular_to_Neo4j
python -m pytest
```

To run tests with verbose output:

```bash
python -m pytest -v
```

To run a specific test file:

```bash
python -m pytest tests/nodes/input/test_csv_loader.py
```

To run a specific test:

```bash
python -m pytest tests/nodes/input/test_csv_loader.py::test_load_csv_node_success
```

### Using tox

Tox is configured to run the tests in multiple Python environments and also run linting checks.

To run all tests with tox:

```bash
cd /app/Tabular_to_Neo4j
tox
```

To run only the tests in a specific Python version:

```bash
tox -e py38  # Python 3.8
tox -e py39  # Python 3.9
tox -e py310  # Python 3.10
```

To run only the linting checks:

```bash
tox -e lint
```

## Test Categories

Tests are marked with the following categories:

- `unit`: Unit tests that test a single component in isolation
- `integration`: Integration tests that test multiple components together
- `slow`: Tests that take a long time to run
- `llm`: Tests that require LLM access

To run tests with a specific marker:

```bash
python -m pytest -m unit
python -m pytest -m integration
```

## Mock Data

The tests use mock data defined in `conftest.py`. This includes:

- Sample CSV data
- Mock LLM responses
- Fixtures for common test objects

## Adding New Tests

When adding new tests:

1. Create a new test file in the appropriate directory
2. Import the necessary components and fixtures
3. Write test functions with descriptive names
4. Use appropriate markers for your tests
5. Mock external dependencies (like LLM calls)

## Continuous Integration

These tests are designed to be run in a CI/CD pipeline. The tox configuration ensures that the tests can be run in multiple Python environments.
