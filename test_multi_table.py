import sys
import os
import logging

# Ensure only /app (project root) is in sys.path for package imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Remove Tabular_to_Neo4j from sys.path if present
tabular_neo4j_path = os.path.join(project_root, "Tabular_to_Neo4j")
if tabular_neo4j_path in sys.path:
    sys.path.remove(tabular_neo4j_path)

print("Python executable:", sys.executable)
print("sys.path:", sys.path)

from Tabular_to_Neo4j.utils import output_saver
from Tabular_to_Neo4j.utils.result_utils import validate_input_path
from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver

initialize_output_saver("samples")

from Tabular_to_Neo4j.utils.output_saver import output_saver
from Tabular_to_Neo4j.graphs.multi_table_graph import run_multi_table_pipeline

# Now run your pipeline
def debug_run_multi_table(
    input_path: str = "/app/Tabular_to_Neo4j/sample_data/csv",  # Now relative to /app
    debug: bool = False,
):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    validate_input_path(input_path, pipeline="multi_table_graph")
    print(f"[DEBUG] Starting multi-table pipeline on: {input_path}")

    try:
        final_state = run_multi_table_pipeline(input_path)
        print("[DEBUG] Pipeline completed successfully.")
        return final_state
    except Exception as e:
        print(f"[DEBUG] Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    debug_run_multi_table()