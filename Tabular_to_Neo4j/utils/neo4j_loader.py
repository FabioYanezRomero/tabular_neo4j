import os
import subprocess
from Tabular_to_Neo4j.utils.output_saver import get_output_saver

def run_neo4j_loader(cypher_json_filename=None):
    """
    Run the loader script to push Cypher queries to Neo4j.
    If no filename is provided, use the latest cypher output in the current run's sample directory.
    """
    output_saver = get_output_saver()
    if output_saver is None:
        raise RuntimeError("OutputSaver not initialized. Cannot locate cypher output.")

    timestamp = output_saver.timestamp
    base_dir = output_saver.base_dir
    cypher_dir = os.path.join(base_dir, timestamp, "node_outputs")

    if cypher_json_filename is None:
        cypher_json_filename = "13_generate_cypher_templates.json"

    cypher_json_path = os.path.join(cypher_dir, cypher_json_filename)
    if not os.path.exists(cypher_json_path):
        raise FileNotFoundError(f"Cypher JSON file not found: {cypher_json_path}")
    loader_script = "/app/scripts/load_cypher.py"
    cmd = ["python", loader_script, cypher_json_path]
    print(f"[Neo4j Loader] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Neo4j Loader] Error: {result.stderr}")
        raise RuntimeError(f"Neo4j loader failed: {result.stderr}")
    print(f"[Neo4j Loader] Output: {result.stdout}")
    return result.stdout
