"""
Main script for the Tabular to Neo4j converter using LangGraph.
"""

import argparse
import json
import os
from typing import Dict, Any
from pathlib import Path
import logging
import sys

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Loaded environment variables from .env file")
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env loading")
except Exception as e:
    logging.warning(f"Failed to load .env file: {e}")


from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from app_state import GraphState
from nodes.input_nodes import load_csv_node, detect_header_heuristic_node
from nodes.header_nodes import (
    infer_header_llm_node, 
    validate_header_llm_node, 
    translate_header_llm_node, 
    apply_header_node
)
from nodes.analysis_nodes import (
    perform_column_analytics_node, 
    llm_semantic_column_analysis_node
)
from nodes.synthesis_nodes import synthesize_schema_node

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_graph() -> StateGraph:
    """
    Create the LangGraph for CSV analysis and Neo4j schema inference.
    
    Returns:
        StateGraph instance
    """
    # Create the graph with the GraphState type
    graph = StateGraph(GraphState)
    
    # Add nodes to the graph
    graph.add_node("load_csv", load_csv_node)
    graph.add_node("detect_header", detect_header_heuristic_node)
    graph.add_node("infer_header", infer_header_llm_node)
    graph.add_node("validate_header", validate_header_llm_node)
    graph.add_node("translate_header", translate_header_llm_node)
    graph.add_node("apply_header", apply_header_node)
    graph.add_node("analyze_columns", perform_column_analytics_node)
    graph.add_node("semantic_analysis", llm_semantic_column_analysis_node)
    graph.add_node("synthesize_schema", synthesize_schema_node)
    
    # Define the edges
    # Start with loading the CSV
    graph.add_edge("load_csv", "detect_header")
    
    # Conditional edge from detect_header based on has_header_heuristic
    graph.add_conditional_edges(
        "detect_header",
        lambda state: "has_header" if state.get("has_header_heuristic", False) else "no_header",
        {
            "has_header": "validate_header",  # If header detected, skip inference
            "no_header": "infer_header"       # If no header detected, infer headers
        }
    )
    
    # Continue the flow from infer_header to validate_header
    graph.add_edge("infer_header", "validate_header")
    
    # Continue with header validation and translation
    graph.add_edge("validate_header", "translate_header")
    graph.add_edge("translate_header", "apply_header")
    
    # Column analysis
    graph.add_edge("apply_header", "analyze_columns")
    graph.add_edge("analyze_columns", "semantic_analysis")
    graph.add_edge("semantic_analysis", "synthesize_schema")
    
    # End the graph after schema synthesis
    graph.add_edge("synthesize_schema", END)
    
    # Set the entry point
    graph.set_entry_point("load_csv")
    
    return graph

def format_schema_output(schema: Dict[str, Any]) -> str:
    """
    Format the inferred Neo4j schema for human-readable output.
    
    Args:
        schema: The inferred Neo4j schema
        
    Returns:
        Formatted string representation
    """
    output = []
    
    # Primary entity
    output.append(f"PRIMARY ENTITY: :{schema['primary_entity_label']}")
    output.append("")
    
    # Group columns by role
    columns_by_role = {}
    for col in schema.get('columns_classification', []):
        role = col.get('role', 'UNKNOWN')
        if role not in columns_by_role:
            columns_by_role[role] = []
        columns_by_role[role].append(col)
    
    # Format primary entity identifiers
    if 'PRIMARY_ENTITY_IDENTIFIER' in columns_by_role:
        output.append("PRIMARY ENTITY IDENTIFIERS:")
        for col in columns_by_role['PRIMARY_ENTITY_IDENTIFIER']:
            output.append(f"  - {col['original_column_name']} → .{col['neo4j_property_key']}")
        output.append("")
    
    # Format primary entity properties
    if 'PRIMARY_ENTITY_PROPERTY' in columns_by_role:
        output.append("PRIMARY ENTITY PROPERTIES:")
        for col in columns_by_role['PRIMARY_ENTITY_PROPERTY']:
            note = f" (Note: {col['note']})" if 'note' in col else ""
            output.append(f"  - {col['original_column_name']} → .{col['neo4j_property_key']}{note}")
        output.append("")
    
    # Format new node types
    if 'NEW_NODE_TYPE_VALUES' in columns_by_role:
        output.append("NEW NODE TYPES:")
        for col in columns_by_role['NEW_NODE_TYPE_VALUES']:
            output.append(f"  - {col['original_column_name']} → :{col['new_node_label']} nodes")
            output.append(f"    Relationship: (:{schema['primary_entity_label']})-[:{col['relationship_to_primary']}]->(:{col['new_node_label']})")
            
            # Add associated properties for this node type
            if 'NEW_NODE_PROPERTY' in columns_by_role:
                associated_props = [
                    prop for prop in columns_by_role['NEW_NODE_PROPERTY']
                    if prop.get('associated_new_node_label') == col['new_node_label']
                ]
                if associated_props:
                    output.append(f"    Properties:")
                    for prop in associated_props:
                        output.append(f"      - {prop['original_column_name']} → .{prop['neo4j_property_key']}")
            
            output.append("")
    
    # Format relationship properties
    if 'RELATIONSHIP_PROPERTY' in columns_by_role:
        output.append("RELATIONSHIP PROPERTIES:")
        for col in columns_by_role['RELATIONSHIP_PROPERTY']:
            rel = col.get('associated_relationship', 'UNKNOWN')
            source = col.get('source_node_label', schema['primary_entity_label'])
            target = col.get('target_node_label', 'UNKNOWN')
            output.append(f"  - {col['original_column_name']} → property on relationship (:{source})-[:{rel}]->(:{target})")
        output.append("")
    
    return "\n".join(output)

def run_analysis(csv_file_path: str, output_file: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Run the CSV analysis and Neo4j schema inference.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file: Optional path to save the results
        verbose: Whether to print verbose output
        
    Returns:
        The final graph state
    """
    # Create the graph
    graph = create_graph()
    
    # Compile the graph
    app = graph.compile()
    
    # Set up the initial state
    initial_state = GraphState(
        csv_file_path=csv_file_path,
        error_messages=[]
    )
    
    # Run the graph
    logger.info(f"Starting analysis of {csv_file_path}")
    final_state = app.invoke(initial_state)
    
    # Print the results
    if verbose:
        logger.info("Analysis complete.")
        
        # Print any errors
        if final_state.get('error_messages'):
            logger.warning("Errors encountered during analysis:")
            for error in final_state.get('error_messages', []):
                logger.warning(f"  - {error}")
        
        # Print the inferred schema
        if final_state.get('inferred_neo4j_schema'):
            schema_output = format_schema_output(final_state['inferred_neo4j_schema'])
            print("\nINFERRED NEO4J SCHEMA:")
            print(schema_output)
    
    # Save the results to a file if specified
    if output_file:
        output_path = Path(output_file)
        
        # Create the output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the results
        with open(output_path, 'w') as f:
            # Extract the relevant parts of the state to save
            results = {
                'inferred_neo4j_schema': final_state.get('inferred_neo4j_schema'),
                'error_messages': final_state.get('error_messages', [])
            }
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return final_state

def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description='Analyze a CSV file and infer a Neo4j schema.')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('--output', '-o', help='Path to save the results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    
    run_analysis(args.csv_file, args.output, args.verbose)

if __name__ == "__main__":
    main()
