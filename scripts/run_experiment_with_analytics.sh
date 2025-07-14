#!/usr/bin/env bash
# Run the multi-table experiment that uses full JSON analytics.
# Usage: ./run_experiment_with_analytics.sh <path_to_dataset_folder> [output_dir] [...additional main.py flags]
set -euo pipefail
INPUT_PATH=${1:-/app/datasets/csvs}
OUTPUT_DIR=${2:-samples}
shift || true  # consume INPUT_PATH
[ "$#" -gt 0 ] && shift || true  # consume OUTPUT_DIR if provided
python -m Tabular_to_Neo4j.main --input_path "$INPUT_PATH" --output_dir "$OUTPUT_DIR" --pipeline experiments_with_analytics "$@"
