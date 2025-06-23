import os
import subprocess
import time
import json

CSV_PATH = "/app/Tabular_to_Neo4j/sample_data/csv/customers.csv"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def test_pipeline_and_neo4j_import():
    """
    Integration test: Run the pipeline and verify Cypher import into Neo4j.
    """
    # Clean up previous samples
    subprocess.run(["rm", "-rf", "/app/samples"], check=True)
    # Run the pipeline (will auto-load Cypher)
    result = subprocess.run([
        "python", "-m", "Tabular_to_Neo4j.main", CSV_PATH, "--save-node-outputs"
    ], capture_output=True, text=True)
    print(result.stdout)
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    # Wait for Neo4j to finish loading
    time.sleep(5)

    # Check Neo4j for imported nodes
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS node_count")
        node_count = result.single()["node_count"]
        print(f"Node count in Neo4j: {node_count}")
        assert node_count > 0, "No nodes imported into Neo4j!"
    driver.close()

if __name__ == "__main__":
    test_pipeline_and_neo4j_import()
    print("Integration test passed!")
