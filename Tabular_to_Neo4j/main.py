"""
Main script for the Tabular to Neo4j converter using LangGraph.
"""

import argparse
import json
import os
import time
import logging
from typing import Dict, Any
from pathlib import Path
from Tabular_to_Neo4j.utils.logging_config import get_logger, setup_logging
import sys

# Initialize logging with default configuration
setup_logging()

# Get a logger for this module
logger = get_logger(__name__)

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")


from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from Tabular_to_Neo4j.app_state import GraphState

# Import input nodes
from Tabular_to_Neo4j.nodes.input import (
    load_csv_node,
    detect_header_heuristic_node
)

# Import header processing nodes
from Tabular_to_Neo4j.nodes.header_processing import (
    infer_header_llm_node, 
    validate_header_llm_node, 
    detect_header_language_node,
    translate_header_llm_node, 
    apply_header_node
)

# Import analysis nodes
from Tabular_to_Neo4j.nodes.analysis import (
    perform_column_analytics_node
)

# Import entity inference nodes
from Tabular_to_Neo4j.nodes.entity_inference import (
    classify_entities_properties_node,
    reconcile_entity_property_node,
    map_properties_to_entities_node,
    infer_entity_relationships_node
)

# Import database schema generation nodes
from Tabular_to_Neo4j.nodes.db_schema import (
    generate_cypher_templates_node
)

# Import output saver
from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver, get_output_saver

# Get a logger for this module
logger = get_logger(__name__)

def create_graph() -> StateGraph:
    """
    Create the LangGraph for CSV analysis and Neo4j schema inference.
    
    Returns:
        StateGraph instance
    """
    # Create the graph with the GraphState type
    graph = StateGraph(GraphState)
    
    # Get the output saver
    output_saver = get_output_saver()
    
    # Define node order for output file naming
    node_order = {
        "load_csv": 1,
        "detect_header": 2,
        "infer_header": 3,
        "validate_header": 4,
        "detect_header_language": 5,
        "translate_header": 6,
        "apply_header": 7,
        "analyze_columns": 8,
        "classify_entities_properties": 9,
        "reconcile_entity_property": 10,
        "map_properties_to_entities": 11,
        "infer_entity_relationships": 12,
        "generate_cypher_templates": 13
    }
    
    # Add nodes to the graph

    # Input nodes
    graph.add_node("load_csv", wrap_node_with_output_saving("load_csv", load_csv_node, node_order["load_csv"]))
    graph.add_node("detect_header", wrap_node_with_output_saving("detect_header", detect_header_heuristic_node, node_order["detect_header"]))
    
    # Header processing nodes
    graph.add_node("infer_header", wrap_node_with_output_saving("infer_header", infer_header_llm_node, node_order["infer_header"]))
    graph.add_node("validate_header", wrap_node_with_output_saving("validate_header", validate_header_llm_node, node_order["validate_header"]))
    graph.add_node("detect_header_language", wrap_node_with_output_saving("detect_header_language", detect_header_language_node, node_order["detect_header_language"]))
    graph.add_node("translate_header", wrap_node_with_output_saving("translate_header", translate_header_llm_node, node_order["translate_header"]))
    graph.add_node("apply_header", wrap_node_with_output_saving("apply_header", apply_header_node, node_order["apply_header"]))
    
    # Analysis nodes
    graph.add_node("analyze_columns", wrap_node_with_output_saving("analyze_columns", perform_column_analytics_node, node_order["analyze_columns"]))
    
    
    # Entity and Relationship Inference
    graph.add_node("classify_entities_properties", wrap_node_with_output_saving("classify_entities_properties", classify_entities_properties_node, node_order["classify_entities_properties"]))
    graph.add_node("reconcile_entity_property", wrap_node_with_output_saving("reconcile_entity_property", reconcile_entity_property_node, node_order["reconcile_entity_property"]))
    graph.add_node("map_properties_to_entities", wrap_node_with_output_saving("map_properties_to_entities", map_properties_to_entities_node, node_order["map_properties_to_entities"]))
    graph.add_node("infer_entity_relationships", wrap_node_with_output_saving("infer_entity_relationships", infer_entity_relationships_node, node_order["infer_entity_relationships"]))
    
    # Database Schema Generation
    graph.add_node("generate_cypher_templates", wrap_node_with_output_saving("generate_cypher_templates", generate_cypher_templates_node, node_order["generate_cypher_templates"]))
    
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
    
    # Add language detection after header validation
    graph.add_edge("validate_header", "detect_header_language")
    
    # Conditional edge from detect_header_language based on is_header_in_target_language
    graph.add_conditional_edges(
        "detect_header_language",
        lambda state: "same_language" if state.get("is_header_in_target_language", False) else "different_language",
        {
            "same_language": "apply_header",       # If header already in target language, skip translation
            "different_language": "translate_header" # If header in different language, translate it
        }
    )
    
    # Continue with header translation (if needed) and application
    graph.add_edge("translate_header", "apply_header")
    
    # Column analysis
    graph.add_edge("apply_header", "analyze_columns")
    # Schema synthesis pipeline
    graph.add_edge("analyze_columns", "classify_entities_properties")
    graph.add_edge("classify_entities_properties", "reconcile_entity_property")
    graph.add_edge("reconcile_entity_property", "map_properties_to_entities")
    graph.add_edge("map_properties_to_entities", "infer_entity_relationships")
    graph.add_edge("infer_entity_relationships", "generate_cypher_templates")
    
    # End the graph after cypher template generation
    graph.add_edge("generate_cypher_templates", END)
    
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

def run_analysis(csv_file_path: str, output_file: str = None, verbose: bool = False, 
               save_node_outputs: bool = False, output_dir: str = "samples") -> Dict[str, Any]:
    """
    Run the CSV analysis and Neo4j schema inference.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file: Optional path to save the results
        verbose: Whether to print verbose output
        
    Returns:
        The final graph state
    """
    # Log analysis start with file details
    logger.info(f"Starting analysis of CSV file: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Log file size and basic info
    file_size = os.path.getsize(csv_file_path) / 1024  # KB
    logger.info(f"File size: {file_size:.2f} KB")
    
    # Initialize output saver if requested
    if save_node_outputs:
        logger.info(f"Initializing output saver with directory: {output_dir}")
        initialize_output_saver(output_dir)
    
    # Create the graph
    logger.debug("Creating state graph")
    graph = create_graph()
    
    # Compile the graph
    logger.debug("Compiling state graph")
    app = graph.compile()
    
    # Set up the initial state
    logger.debug("Initializing state")
    initial_state = GraphState(
        csv_file_path=csv_file_path,
        error_messages=[]
    )
    
    # Run the graph
    logger.info(f"Executing analysis pipeline")
    start_time = time.time()
    try:
        final_state = app.invoke(initial_state)
        execution_time = time.time() - start_time
        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise
    
    # Log the results
    if final_state.get('error_messages'):
        error_count = len(final_state.get('error_messages', []))
        logger.warning(f"Analysis completed with {error_count} errors/warnings")
        for i, error in enumerate(final_state.get('error_messages', [])):
            logger.warning(f"Error/Warning {i+1}: {error}")
    else:
        logger.info("Analysis completed successfully with no errors")
    
    # Log cypher template information
    if final_state.get('cypher_query_templates'):
        templates = final_state['cypher_query_templates']
        entity_count = len(templates.get('entity_creation_queries', []))
        relationship_count = len(templates.get('relationship_queries', []))
        example_count = len(templates.get('example_queries', []))
        logger.info(f"Generated {entity_count} entity creation queries, {relationship_count} relationship queries, and {example_count} example queries")
    else:
        logger.warning("No Cypher templates were generated")
    
    # Save results to file if requested
    if output_file:
        try:
            logger.info(f"Saving results to {output_file}")
            with open(output_file, 'w') as f:
                if final_state.get('cypher_query_templates'):
                    templates = final_state['cypher_query_templates']
                    f.write("ENTITY CREATION QUERIES:\n")
                    for i, query in enumerate(templates.get('entity_creation_queries', [])):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                    
                    f.write("\nRELATIONSHIP QUERIES:\n")
                    for i, query in enumerate(templates.get('relationship_queries', [])):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                    
                    f.write("\nEXAMPLE QUERIES:\n")
                    for i, query in enumerate(templates.get('example_queries', [])):
                        f.write(f"\nQuery {i+1}:\n{query.get('query', '')}\n")
                else:
                    f.write("No Cypher templates generated.\n")
                    
                if final_state.get('error_messages'):
                    f.write("\nErrors/Warnings:\n")
                    for error in final_state.get('error_messages', []):
                        f.write(f"  - {error}\n")
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {str(e)}")
    
    # Print the results if verbose
    if verbose:
        logger.info("Displaying analysis results")
        
        # Print any errors
        if final_state.get('error_messages'):
            print("\nErrors/Warnings encountered during analysis:")
            for error in final_state.get('error_messages', []):
                print(f"  - {error}")
        
        # Print the generated Cypher templates
        if final_state.get('cypher_query_templates'):
            templates = final_state['cypher_query_templates']
            print("\nGenerated Cypher Templates:")
            
            print("\nENTITY CREATION QUERIES:")
            for i, query in enumerate(templates.get('entity_creation_queries', [])):
                print(f"\nQuery {i+1}:")
                print(query.get('query', ''))
            
            print("\nRELATIONSHIP QUERIES:")
            for i, query in enumerate(templates.get('relationship_queries', [])):
                print(f"\nQuery {i+1}:")
                print(query.get('query', ''))
            
            print("\nEXAMPLE QUERIES:")
            for i, query in enumerate(templates.get('example_queries', [])):
                print(f"\nQuery {i+1}:")
                print(query.get('query', ''))
    
    return final_state

def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description='Run Tabular to Neo4j converter.')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('--output', '-o', help='Path to save the results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--save-node-outputs', '-s', action='store_true', 
                        help='Save the output of each node to files')
    parser.add_argument('--output-dir', '-d', default="samples",
                        help='Directory to save node outputs to (default: samples)')
    
    args = parser.parse_args()
    
    try:
        run_analysis(args.csv_file, args.output, args.verbose, 
                     args.save_node_outputs, args.output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def wrap_node_with_output_saving(node_name: str, node_func, node_order: int = 0):
    """
    Wrap a node function with output saving functionality.
    
    Args:
        node_name: Name of the node
        node_func: Node function to wrap
        node_order: Order of the node in the pipeline (for file naming)
        
    Returns:
        Wrapped node function
    """
    def wrapped_node(state, config=None):
        # Call the original node function with the config parameter
        result = node_func(state, config) if config is not None else node_func(state)
        
        # Save the output if output saver is initialized
        output_saver = get_output_saver()
        if output_saver:
            output_saver.save_node_output(node_name, result, node_order)
        
        return result
    
    return wrapped_node

if __name__ == "__main__":
    main()
