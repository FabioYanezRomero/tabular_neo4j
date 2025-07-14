#!/usr/bin/env bash
# Run experiment that uses only column samples (no analytics)
# Usage: ./run_experiment_without_analytics.sh <dataset_folder> [output_dir] [...flags]
set -euo pipefail
INPUT_PATH=${1:-/app/datasets/csvs}
OUTPUT_DIR=${2:-samples}
shift || true
[ "$#" -gt 0 ] && shift || true
python -m Tabular_to_Neo4j.main --input_path "$INPUT_PATH" --output_dir "$OUTPUT_DIR" --pipeline experiments_without_analytics "$@"
