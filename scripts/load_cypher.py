import os
import json
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

IMPORT_DIR = os.getenv("CYPHER_IMPORT_DIR", "/import")


def load_cypher_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Accepts either a list of cypher strings or a dict with 'cypher' field
    if isinstance(data, dict) and 'cypher' in data:
        queries = [data['cypher']] if isinstance(data['cypher'], str) else data['cypher']
    elif isinstance(data, list):
        queries = data
    else:
        raise ValueError("JSON must contain a list of Cypher queries or a dict with a 'cypher' field.")
    return queries


def run_queries(queries):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for query in queries:
            print(f"Running Cypher: {query[:100]}...")
            session.run(query)
    driver.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load Cypher queries from JSON and run them in Neo4j.")
    parser.add_argument("json_file", help="Path to the JSON file with Cypher queries.")
    args = parser.parse_args()
    json_path = args.json_file
    if not os.path.isabs(json_path):
        json_path = os.path.join(IMPORT_DIR, json_path)
    queries = load_cypher_from_json(json_path)
    run_queries(queries)

if __name__ == "__main__":
    main()
