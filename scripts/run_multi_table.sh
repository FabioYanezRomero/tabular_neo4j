#!/bin/bash
# Script to run the Tabular_to_Neo4j pipeline for multi-table (multi-CSV) workflows

set -e

# Paths to sample data (folder containing multiple CSVs)
CSV_FOLDER="/app/Tabular_to_Neo4j/sample_data/csv"

LOG_LEVEL="INFO"

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: ./scripts/run_multi_table.sh [CSV_FOLDER] [PIPELINE] [--save-node-outputs] [--log-level LEVEL]"
    echo "Default CSV_FOLDER: $CSV_FOLDER"
    echo "Default PIPELINE: multi_table_graph"
    echo "  PIPELINE            Name of the graph/pipeline to use (e.g. multi_table_graph)"
    echo "  --log-level LEVEL   Set Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO."
    exit 0
fi

if [ -n "$1" ] && [[ "$1" != --* ]]; then
    CSV_FOLDER="$1"
    shift
fi

# Pipeline/graph selection
table_pipeline="multi_table_graph"
if [ -n "$1" ] && [[ "$1" != --* ]]; then
    table_pipeline="$1"
    shift
fi

EXTRA_ARGS=""
for arg in "$@"; do
    if [[ "$arg" == --save-node-outputs ]]; then
        EXTRA_ARGS="--save-node-outputs"
    fi
    if [[ "$arg" == --log-level* ]]; then
        LOG_LEVEL="${arg#--log-level=}"
    fi
    if [[ "$arg" == --log-level ]]; then
        shift
        LOG_LEVEL="$1"
    fi
    shift
    if [ -z "$1" ]; then break; fi
    set -- "$@"

done

echo "[INFO] Running Tabular_to_Neo4j multi-table pipeline on all CSVs in $CSV_FOLDER using pipeline $table_pipeline ..."

# Run the multi-table pipeline directly on the folder of CSVs using multi_table_graph.py
if [ ! -d "$CSV_FOLDER" ]; then
    echo "[ERROR] CSV folder $CSV_FOLDER does not exist."
    exit 1
fi

# Run the multi-table pipeline
python3 /app/Tabular_to_Neo4j/graphs/multi_table_graph.py --csv-folder "$CSV_FOLDER" --pipeline "$table_pipeline" --log-level "$LOG_LEVEL" $EXTRA_ARGS
