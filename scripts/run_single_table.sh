#!/bin/bash
# Script to run the full Tabular_to_Neo4j pipeline on the sample customers.csv and customers.json

set -e

# Paths to sample data
CSV_FOLDER="/app/Tabular_to_Neo4j/sample_data/csv"

LOG_LEVEL="DEBUG"

# Allow user to override CSV path or add --save-node-outputs
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_example.sh [CSV_FOLDER] [PIPELINE] [--save-node-outputs] [--log-level LEVEL]"
    echo "Default CSV_FOLDER: $CSV_FOLDER"
    echo "Default PIPELINE: single_table_graph"
    echo "  PIPELINE            Name of the graph/pipeline to use (e.g. single_table_graph)"
    echo "  --log-level LEVEL   Set Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO."
    exit 0
fi

if [ -n "$1" ] && [[ "$1" != --* ]]; then
    CSV_FOLDER="$1"
    shift
fi

# Find the first CSV file in the folder
CSV_PATH=$(find "$CSV_FOLDER" -maxdepth 1 -type f -name '*.csv' | head -n 1)
if [ -z "$CSV_PATH" ]; then
    echo "[ERROR] No CSV file found in folder: $CSV_FOLDER"
    exit 1
fi

# Pipeline/graph selection
PIPELINE="single_table_graph"
if [ -n "$1" ] && [[ "$1" != --* ]]; then
    PIPELINE="$1"
    shift
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] CSV file not found: $CSV_PATH"
    exit 1
fi

EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" == --save-node-outputs ]]; then
        EXTRA_ARGS="--save-node-outputs"
    fi
    if [[ "$arg" == --log-level* ]]; then
        LOG_LEVEL="${arg#--log-level=}" # --log-level=DEBUG or --log-level DEBUG
    fi
    if [[ "$arg" == --log-level ]]; then
        shift
        LOG_LEVEL="$1"
    fi
    shift
    if [ -z "$1" ]; then break; fi
    set -- "$@"
done

echo "[INFO] Running Tabular_to_Neo4j pipeline on $CSV_PATH using pipeline $PIPELINE ..."

# Detect the default LLM provider from settings.py
LLM_PROVIDER=$(python -c "import sys; sys.path.insert(0, '/app'); from Tabular_to_Neo4j.config.settings import DEFAULT_LLM_PROVIDER; print(DEFAULT_LLM_PROVIDER)")

if [ "$LLM_PROVIDER" = "ollama" ]; then
    python3 -m Tabular_to_Neo4j.main --input_path "$CSV_PATH" --pipeline "$PIPELINE" --log-level "$LOG_LEVEL" $EXTRA_ARGS
else
    python3 -m Tabular_to_Neo4j.main --input_path "$CSV_PATH" --pipeline "$PIPELINE" --log-level "$LOG_LEVEL" $EXTRA_ARGS
fi

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Pipeline run complete. Check Neo4j or output directories for results."
else
    echo "[FAILURE] Pipeline run failed. See output above."
    exit 2
fi
