#!/usr/bin/env python3
"""
Script to run the Tabular to Neo4j converter with LMStudio integration.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the repository root to the Python path if needed
repo_root = Path(__file__).parent.absolute()
if str(repo_root.parent) not in sys.path:
    sys.path.insert(0, str(repo_root.parent))

from Tabular_to_Neo4j.utils.check_lmstudio import check_lmstudio_connection
from Tabular_to_Neo4j.main import run_analysis

def main():
    """
    Main entry point for running with LMStudio integration.
    """
    parser = argparse.ArgumentParser(description='Run Tabular to Neo4j converter with LMStudio integration.')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('--output', '-o', help='Path to save the results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--lmstudio-host', default=os.environ.get('LMSTUDIO_HOST', 'host.docker.internal'), 
                      help='LMStudio host address (default: from env or host.docker.internal)')
    parser.add_argument('--lmstudio-port', type=int, default=int(os.environ.get('LMSTUDIO_PORT', 1234)), 
                      help='LMStudio port number (default: from env or 1234)')
    parser.add_argument('--retries', type=int, default=3, help='Number of connection retries')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        print(f"   Please check the path and try again.")
        sys.exit(1)
    
    # Check if LMStudio is reachable
    print(f"Checking LMStudio connection at {args.lmstudio_host}:{args.lmstudio_port}...")
    lmstudio_available = check_lmstudio_connection(
        host=args.lmstudio_host,
        port=args.lmstudio_port,
        retries=args.retries
    )
    
    if not lmstudio_available:
        print("❌ LMStudio is not available. Please make sure LMStudio is running and properly configured.")
        print("   You can start LMStudio and make sure it's listening on the specified host and port.")
        print("   If running in Docker, make sure the container has access to the host network.")
        sys.exit(1)
    
    # Update the LMStudio configuration
    config_dir = os.path.join(repo_root, "config")
    config_file = os.path.join(config_dir, "lmstudio_config.py")
    
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_content = f.read()
        
        # Update the host and port
        config_content = config_content.replace(
            f'LMSTUDIO_HOST = "host.docker.internal"',
            f'LMSTUDIO_HOST = "{args.lmstudio_host}"'
        )
        config_content = config_content.replace(
            f'LMSTUDIO_PORT = 1234',
            f'LMSTUDIO_PORT = {args.lmstudio_port}'
        )
        
        with open(config_file, "w") as f:
            f.write(config_content)
    
    # Run the analysis
    print(f"Running analysis on {args.csv_file} with LMStudio integration...")
    try:
        final_state = run_analysis(args.csv_file, args.output, args.verbose)
        print("✅ Analysis completed successfully!")
        
        # Check if Cypher templates were generated
        if final_state.get('cypher_query_templates'):
            templates = final_state['cypher_query_templates']
            entity_count = len(templates.get('entity_creation_queries', []))
            relationship_count = len(templates.get('relationship_queries', []))
            print(f"Generated {entity_count} entity creation queries and {relationship_count} relationship queries.")
        else:
            print("⚠️ No Cypher templates were generated.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
