#!/bin/bash
# Script to run the full Tabular_to_Neo4j pipeline on the sample customers.csv and customers.json

set -e

# Paths to sample data
CSV_PATH="/app/Tabular_to_Neo4j/sample_data/csv/customers.csv"

LOG_LEVEL="DEBUG"

# Allow user to override CSV path or add --save-node-outputs
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_example.sh [CSV_PATH] [--save-node-outputs] [--log-level LEVEL]"
    echo "Default CSV_PATH: $CSV_PATH"
    echo "  --log-level LEVEL   Set Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO."
    exit 0
fi

if [ -n "$1" ] && [[ "$1" != --* ]]; then
    CSV_PATH="$1"
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

echo "[INFO] Running Tabular_to_Neo4j pipeline on $CSV_PATH ..."

# Detect the default LLM provider from settings.py
LLM_PROVIDER=$(python -c "import sys; sys.path.insert(0, '/app'); from Tabular_to_Neo4j.config.settings import DEFAULT_LLM_PROVIDER; print(DEFAULT_LLM_PROVIDER)")

if [ "$LLM_PROVIDER" = "ollama" ]; then
    LOG_LEVEL="$LOG_LEVEL" python -m Tabular_to_Neo4j.run_with_ollama "$CSV_PATH" $EXTRA_ARGS
else
    LOG_LEVEL="$LOG_LEVEL" python -m Tabular_to_Neo4j.run_with_lmstudio "$CSV_PATH" $EXTRA_ARGS
fi

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Pipeline run complete. Check Neo4j or output directories for results."
else
    echo "[FAILURE] Pipeline run failed. See output above."
    exit 2
fi
