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
    perform_column_analytics_node, 
    llm_semantic_column_analysis_node
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
    generate_cypher_templates_node,
    synthesize_final_schema_node
)

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
    
    # Add nodes to the graph

    # Input nodes
    graph.add_node("load_csv", load_csv_node)
    graph.add_node("detect_header", detect_header_heuristic_node)
    
    # Header processing nodes
    graph.add_node("infer_header", infer_header_llm_node)
    graph.add_node("validate_header", validate_header_llm_node)
    graph.add_node("detect_header_language", detect_header_language_node)
    graph.add_node("translate_header", translate_header_llm_node)
    graph.add_node("apply_header", apply_header_node)
    
    # Analysis nodes
    graph.add_node("analyze_columns", perform_column_analytics_node)
    graph.add_node("semantic_analysis", llm_semantic_column_analysis_node)
    
    
    # Schema synthesis nodes - Group 1: Entity and Relationship Inference
    # These nodes analyze the data to identify entities, properties, and their relationships
    graph.add_node("classify_entities_properties", classify_entities_properties_node)
    graph.add_node("reconcile_entity_property", reconcile_entity_property_node)
    graph.add_node("map_properties_to_entities", map_properties_to_entities_node)
    graph.add_node("infer_entity_relationships", infer_entity_relationships_node)
    
    # Schema synthesis nodes - Group 2: Database Schema Generation
    # These nodes generate database-specific artifacts like Cypher templates and final schema
    graph.add_node("generate_cypher_templates", generate_cypher_templates_node)
    graph.add_node("synthesize_final_schema", synthesize_final_schema_node)
    
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
    graph.add_edge("analyze_columns", "semantic_analysis")
    # Schema synthesis pipeline
    graph.add_edge("semantic_analysis", "classify_entities_properties")
    graph.add_edge("classify_entities_properties", "reconcile_entity_property")
    graph.add_edge("reconcile_entity_property", "map_properties_to_entities")
    graph.add_edge("map_properties_to_entities", "infer_entity_relationships")
    graph.add_edge("infer_entity_relationships", "generate_cypher_templates")
    graph.add_edge("generate_cypher_templates", "synthesize_final_schema")
    
    # End the graph after schema synthesis
    graph.add_edge("synthesize_final_schema", END)
    
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
    # Log analysis start with file details
    logger.info(f"Starting analysis of CSV file: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Log file size and basic info
    file_size = os.path.getsize(csv_file_path) / 1024  # KB
    logger.info(f"File size: {file_size:.2f} KB")
    
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
    
    # Log schema information
    if final_state.get('inferred_neo4j_schema'):
        schema = final_state['inferred_neo4j_schema']
        primary_entity = schema.get('primary_entity_label', 'Unknown')
        column_count = len(schema.get('columns_classification', []))
        logger.info(f"Inferred schema with primary entity '{primary_entity}' and {column_count} classified columns")
        
        # Count column roles
        roles = {}
        for col in schema.get('columns_classification', []):
            role = col.get('role', 'UNKNOWN')
            roles[role] = roles.get(role, 0) + 1
        
        for role, count in roles.items():
            logger.debug(f"Role '{role}': {count} columns")
    else:
        logger.warning("No schema was inferred")
    
    # Save results to file if requested
    if output_file:
        try:
            logger.info(f"Saving results to {output_file}")
            with open(output_file, 'w') as f:
                if final_state.get('inferred_neo4j_schema'):
                    schema_output = format_schema_output(final_state['inferred_neo4j_schema'])
                    f.write(schema_output)
                else:
                    f.write("No schema inferred.\n")
                    
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
        
        # Print the inferred schema
        if final_state.get('inferred_neo4j_schema'):
            schema_output = format_schema_output(final_state['inferred_neo4j_schema'])
            print("\nInferred Neo4j Schema:")
            print(schema_output)
    
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
