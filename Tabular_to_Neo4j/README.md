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

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tabular_neo4j.git
cd tabular_neo4j/Tabular_to_Neo4j
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your LLM provider in `config.py`:
```python
# Set your API key here or use environment variables
LLM_API_KEY = "your-api-key"
# Choose your LLM provider: "openai", "ollama", "anthropic", "lmstudio", "huggingface"
LLM_PROVIDER = "lmstudio"  # Using LMStudio for GGUF models
```

## LMStudio Integration

This project is designed to work with LMStudio for using GGUF format models. Each state in the processing pipeline can use a different model, with automatic loading and unloading of models as needed.

### Setting Up LMStudio

1. **Install LMStudio**:
   - Download and install LMStudio from [lmstudio.ai](https://lmstudio.ai/)
   - Import your GGUF models into LMStudio via the Models tab

2. **Start the LMStudio Server**:
   - In LMStudio, go to the "Server" tab
   - Click "Start Server" to start the API server on port 1234
   - Make sure the server is running before executing your Python code

3. **Configure Models in `config.py`**:
   - Update the `LLM_CONFIGS` dictionary to match your models in LMStudio
   - **Important**: The model names must match exactly with what's in LMStudio
   - Example configuration:

```python
LLM_CONFIGS = {
    "infer_header": {
        "provider": "lmstudio",
        "model_name": "Mistral-7B-Instruct-v0.2-GGUF",  # Must match exactly with LMStudio
        "temperature": 0.0,
        "seed": 42,
        "description": "Model for inferring headers when none are detected",
        "auto_load": True,  # Load model when state starts
        "auto_unload": True  # Unload model when state ends
    },
    # ... other state configurations
}
```

### How the Integration Works

1. **Model Management**:
   - When a state starts (e.g., `infer_header`), its designated model is loaded in LMStudio
   - When the state completes, the model is unloaded to free up resources
   - The system tracks which models are loaded and for how long

2. **Reproducibility**:
   - All models use a fixed seed (default: 42) for reproducible results
   - Temperature is set to 0.0 for deterministic outputs
   - Models operate in a zero-shot setup

3. **Output Formatting**:
   - Each state specifies the expected output format for its model
   - Format instructions are automatically added to prompts

### Troubleshooting

- **Model Not Found**: Ensure the model name in `config.py` matches exactly with LMStudio
- **Server Connection**: Verify LMStudio server is running on port 1234
- **Memory Issues**: If you experience out-of-memory errors, try setting `auto_unload: True` for all states
- **Logs**: Check terminal output for detailed logs about model loading/unloading

### Testing the Connection

Run this script to verify your LMStudio connection:

```python
from Tabular_to_Neo4j.utils.llm_manager import get_lmstudio_models, list_loaded_models

# Get available models from LMStudio
models = get_lmstudio_models()
print("Available models in LMStudio:", [model.get("name") for model in models])

# Run your analysis
# ... your code here ...

# Check which models were loaded
loaded_models = list_loaded_models()
for model in loaded_models:
    print(f"Model: {model['name']}")
    print(f"  Loaded: {model['is_loaded']}")
    print(f"  Last used by state: {model['last_state']}")
```

## Usage

### Command Line Interface

```bash
python main.py path/to/your/data.csv --output results.json --verbose
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

## LLM Provider Support

The system supports multiple LLM providers:
- OpenAI
- LMStudio (local inference)
- Hugging Face
- (Easily extensible to other providers)

## Configuration

You can customize the system behavior in `config.py`:

```python
# LLM Configuration
LLM_API_KEY = ""  # Your API key
LLM_PROVIDER = "openai"  # LLM provider
TARGET_HEADER_LANGUAGE = "English"  # Target language for headers

# Analysis Settings
UNIQUENESS_THRESHOLD = 0.9  # Threshold for considering a column as a unique identifier
```

## Project Structure

```
Tabular_to_Neo4j/
├── main.py                 # Main script to run the LangGraph flow
├── app_state.py            # Defines the TypedDict for LangGraph state
├── nodes/                  # Directory for LangGraph node functions
│   ├── input_nodes.py      # Nodes for loading and initial CSV parsing
│   ├── header_nodes.py     # Nodes for header detection, inference, validation, translation
│   ├── analysis_nodes.py   # Nodes for statistical and LLM-based column analysis
│   └── synthesis_nodes.py  # Node for combining results and inferring schema
├── utils/                  # Utility functions
│   ├── csv_utils.py        # CSV parsing helpers
│   ├── llm_utils.py        # LLM interaction helpers
│   └── analytics_utils.py  # Column statistical analysis functions
├── prompts/                # Directory for storing LLM prompt templates
│   ├── infer_header.txt
│   ├── validate_header.txt
│   ├── translate_header.txt
│   └── analyze_column_semantic.txt
├── config.py               # Configuration settings
├── requirements.txt
└── README.md
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
