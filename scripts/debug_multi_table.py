import os
import sys

# Adjust import paths for scripts/ directory context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path:", sys.path)


from Tabular_to_Neo4j.utils.result_utils import validate_input_path
from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver
from Tabular_to_Neo4j.graphs.multi_table_graph import run_multi_table_pipeline



def debug_run_multi_table(
    input_path: str = "/app/Tabular_to_Neo4j/sample_data/csv",  # Relative to scripts/
    output_dir: str = "/app/samples"
):

    abs_input_path = os.path.abspath(input_path)
    print("Absolute input_path:", abs_input_path)
    print("Exists:", os.path.exists(abs_input_path))
    print("Is dir:", os.path.isdir(abs_input_path))

    # Validate input path
    validate_input_path(input_path, pipeline="multi_table_graph")
    print(f"[DEBUG] Starting multi-table pipeline on: {input_path}")

    # Initialize output saver (creates timestamped output dir)
    output_saver = initialize_output_saver(output_dir)
    if not output_saver:
        raise RuntimeError("OutputSaver is not initialized.")

    # Run the multi-table pipeline
    try:
        final_state = run_multi_table_pipeline(input_path)
        print("[DEBUG] Pipeline completed successfully.")
        # Optionally, print or inspect final_state here
        return final_state
    except Exception as e:
        print(f"[DEBUG] Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # You can set breakpoints here
    debug_run_multi_table()
