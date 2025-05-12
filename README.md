# Tabular_to_Neo4j

A Python repository using LangGraph to analyze CSV files and infer Neo4j graph database schemas. This tool helps bridge the gap between tabular data and graph databases by automatically suggesting how to model your CSV data in Neo4j.

## Features

- **Intelligent Header Processing**: Detects, infers, validates, and translates CSV headers
- **Column Analysis**: Performs statistical and pattern-based analysis on each column
- **Semantic Understanding**: Uses LLMs to understand the meaning and relationships in your data
- **Neo4j Schema Inference**: Suggests how to model your data in Neo4j, including:
  - Primary entity identification
  - Property assignment
  - Related entity detection
  - Relationship modeling
- **Modular Architecture**: Well-organized codebase with clear separation of concerns
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Docker Support**: Run the application in an isolated environment with all dependencies

## Installation

### Standard Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tabular_neo4j.git
cd tabular_neo4j
```

2. Install the required dependencies:
```bash
pip install -r Tabular_to_Neo4j/requirements.txt
```

### Docker Installation (Recommended)

The project includes Docker support for easy setup and consistent environment:

1. Make sure you have Docker installed on your system
2. Build the Docker image:
```bash
make build
```

3. Run the container:
```bash
make run
```

4. Access the container shell:
```bash
make shell
```

## Docker Commands

The project includes a Makefile to simplify Docker operations:

- `make build`: Build the Docker image based on Ubuntu 22.04
- `make run`: Run the container in detached mode with volume mounting
- `make shell`: Open a shell in the running container for interactive use
- `make stop`: Stop and remove the running container
- `make clean`: Remove both the container and image
- `make restart`: Rebuild and restart the container

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Input Package
- **csv_loader**: Handles loading CSV files into pandas DataFrames
- **header_detection**: Applies heuristics to detect headers

### Header Processing Package
- **header_inference**: Uses LLM to infer headers when none are detected
- **header_validation**: Validates and improves headers
- **language_detection**: Detects the language of headers
- **header_translation**: Translates headers to the target language
- **header_application**: Applies headers to the DataFrame

### Analysis Package
- **column_analytics**: Performs statistical and pattern analysis on columns
- **semantic_analysis**: Uses LLM to analyze semantic meaning of columns

### Schema Synthesis Package
- **entity_classification**: Classifies columns as entities or properties
- **entity_reconciliation**: Reconciles different classification approaches
- **property_mapping**: Maps properties to their respective entities
- **relationship_inference**: Infers relationships between entity types
- **cypher_generation**: Generates Cypher query templates
- **schema_finalization**: Combines all results into the final Neo4j schema

### Alternative Synthesis Package
- Provides an alternative implementation of schema synthesis

## Usage

### Command Line Interface

```bash
python Tabular_to_Neo4j/main.py path/to/your/data.csv --output results.json --verbose
```

Arguments:
- `csv_file`: Path to the CSV file to analyze
- `--output` or `-o`: Path to save the results as JSON (optional)
- `--verbose` or `-v`: Print verbose output (optional)

### Python API

```python
from Tabular_to_Neo4j.main import run_analysis

# Run the analysis
result = run_analysis('path/to/your/data.csv', output_file='results.json', verbose=True)

# Access the inferred schema
schema = result.get('inferred_neo4j_schema')
```

## How It Works

The system processes your CSV file through a series of steps using LangGraph:

1. **CSV Loading and Header Detection**:
   - Loads the CSV file and determines if it has headers
   - If no headers are found, uses an LLM to infer appropriate headers

2. **Header Processing**:
   - Validates headers for accuracy and descriptiveness
   - Translates headers to the target language if needed

3. **Column Analysis**:
   - Calculates statistical metrics (uniqueness, cardinality, etc.)
   - Detects patterns and data types
   - Uses LLM for semantic understanding of each column

4. **Schema Synthesis**:
   - Identifies primary entity and its properties
   - Detects columns that should become separate nodes
   - Infers relationships between entities
   - Suggests property assignments for nodes and relationships

## LLM Provider

The system exclusively uses LM Studio for all LLM interactions:
- **LM Studio**: Local inference server for running GGUF models
- **Setup Required**: You need to have LM Studio running locally at http://localhost:1234
- **Models**: The application is configured to use various GGUF models for different tasks (Mistral, Llama2, etc.)

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Launch LM Studio and load your preferred GGUF models
3. Start the local inference server (click on 'Local Inference Server' in LM Studio)
4. Ensure it's running on port 1234 (default)

## Configuration

You can customize the system behavior in `config.py`:

```python
# LLM Configuration
LLM_API_KEY = ""  # Not used with LM Studio but kept for compatibility
LLM_PROVIDER = "lmstudio"  # Using LM Studio exclusively
TARGET_HEADER_LANGUAGE = "English"  # Target language for headers

# Analysis Settings
UNIQUENESS_THRESHOLD = 0.9  # Threshold for considering a column as a unique identifier
```

## Project Structure

```
tabular_neo4j/
├── Dockerfile              # Docker configuration for Ubuntu 22.04
├── Makefile                # Commands for Docker management
├── Tabular_to_Neo4j/       # Main application directory
│   ├── main.py             # Main script to run the LangGraph flow
│   ├── app_state.py        # Defines the TypedDict for LangGraph state
│   ├── nodes/              # Directory for LangGraph node functions
│   │   ├── input_nodes.py  # Nodes for loading and initial CSV parsing
│   │   ├── header_nodes.py # Nodes for header detection, inference, validation, translation
│   │   ├── analysis_nodes.py # Nodes for statistical and LLM-based column analysis
│   │   └── synthesis_nodes.py # Node for combining results and inferring schema
│   ├── utils/              # Utility functions
│   │   ├── csv_utils.py    # CSV parsing helpers
│   │   ├── llm_utils.py    # LLM interaction helpers
│   │   └── analytics_utils.py # Column statistical analysis functions
│   ├── prompts/            # Directory for storing LLM prompt templates
│   │   ├── infer_header.txt
│   │   ├── validate_header.txt
│   │   ├── translate_header.txt
│   │   └── analyze_column_semantic.txt
│   ├── config.py           # Configuration settings
│   ├── requirements.txt
│   └── README.md
├── setup.py                # Package setup file
└── setup_env.sh            # Environment setup script
```

## Example Output

```
INFERRED NEO4J SCHEMA:
PRIMARY ENTITY: :Customer

PRIMARY ENTITY IDENTIFIERS:
  - customer_id → .customerId

PRIMARY ENTITY PROPERTIES:
  - name → .name
  - email → .email
  - signup_date → .signupDate

NEW NODE TYPES:
  - city → :City nodes
    Relationship: (:Customer)-[:LIVES_IN]->(:City)
    Properties:
      - population → .population
      - country → .country

  - order_id → :Order nodes
    Relationship: (:Customer)-[:PLACED]->(:Order)
    Properties:
      - order_date → .orderDate
      - total_amount → .totalAmount

RELATIONSHIP PROPERTIES:
  - loyalty_points → property on relationship (:Customer)-[:PLACED]->(:Order)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
