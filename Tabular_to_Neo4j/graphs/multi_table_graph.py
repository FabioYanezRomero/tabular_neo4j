"""
Graph definition for multi-table workflows in Tabular_to_Neo4j.
Each table's state is separated in a dictionary keyed by table name.
This graph processes all tables in a folder up to the `infer_entity_relationships` node.
"""
from langgraph.graph import StateGraph, END
from Tabular_to_Neo4j.app_state import MultiTableGraphState, GraphState
from Tabular_to_Neo4j.nodes.input import load_csv_node, detect_header_heuristic_node
from Tabular_to_Neo4j.nodes.header_processing import (
    infer_header_llm_node,
    validate_header_llm_node,
    detect_header_language_node,
    translate_header_llm_node,
    apply_header_node,
)
from Tabular_to_Neo4j.nodes.intra_table_analysis import perform_column_analytics_node
from Tabular_to_Neo4j.nodes.entity_inference import (
    classify_entities_properties_node,
    reconcile_entity_property_node,
    map_properties_to_entities_node,
    infer_entity_relationships_node,
)
from Tabular_to_Neo4j.nodes.cross_table_analysis.columns_contextualization import columns_contextualization_node
from Tabular_to_Neo4j.nodes.cross_table_analysis.semantic_embedding_node import semantic_embedding_node
from Tabular_to_Neo4j.nodes.cross_table_analysis.llm_relation_node import llm_relation_node
import os
from typing import Dict, Any, Optional
from Tabular_to_Neo4j.utils.metadata_utils import get_metadata_path_for_csv

# Only up to infer_entity_relationships
PIPELINE_NODES = [
    ("load_csv", load_csv_node),
    ("detect_header", detect_header_heuristic_node),
    ("infer_header", infer_header_llm_node),
    ("validate_header", validate_header_llm_node),
    ("detect_header_language", detect_header_language_node),
    ("translate_header", translate_header_llm_node),
    ("apply_header", apply_header_node),
    ("analyze_columns", perform_column_analytics_node),
    ("classify_entities_properties", classify_entities_properties_node),
    ("reconcile_entity_property", reconcile_entity_property_node),
    ("map_properties_to_entities", map_properties_to_entities_node),
    ("infer_entity_relationships", infer_entity_relationships_node),
]

PIPELINE_EDGES = [
    ("load_csv", "detect_header"),
    {
        "source": "detect_header",
        "condition": lambda state: (
            "has_header" if state.get("has_header_heuristic", False) else "no_header"
        ),
        "edges": {"has_header": "validate_header", "no_header": "infer_header"},
    },
    ("infer_header", "validate_header"),
    ("validate_header", "detect_header_language"),
    {
        "source": "detect_header_language",
        "condition": lambda state: (
            "same_language" if state.get("is_header_in_target_language", False) else "different_language"
        ),
        "edges": {"same_language": "apply_header", "different_language": "translate_header"},
    },
    ("translate_header", "apply_header"),
    ("apply_header", "analyze_columns"),
    ("analyze_columns", "classify_entities_properties"),
    ("classify_entities_properties", "reconcile_entity_property"),
    ("reconcile_entity_property", "map_properties_to_entities"),
    ("map_properties_to_entities", "infer_entity_relationships"),
]

ENTRY_POINT = PIPELINE_NODES[0][0]



def create_multi_table_graph() -> StateGraph:
    """
    Returns the pipeline structure (StateGraph) for a single table in the multi-table workflow.
    This is analogous to create_single_table_graph, but for use in multi-table contexts.
    The folder path and table enumeration are handled elsewhere (initialize_multi_table_state and pipeline runner).
    """
    graph = StateGraph(GraphState)
    for node_name, node_func in PIPELINE_NODES:
        graph.add_node(node_name, node_func)
    for edge in PIPELINE_EDGES:
        if isinstance(edge, dict):
            graph.add_conditional_edges(edge["source"], edge["condition"], edge["edges"])
        else:
            graph.add_edge(*edge)
    graph.set_entry_point(ENTRY_POINT)
    graph.set_finish_point("infer_entity_relationships")
    return graph


# Cross-table nodes to be run after all per-table nodes have finished
CROSS_TABLE_NODES = [
    ("columns_contextualization", columns_contextualization_node),
    ("semantic_embedding", semantic_embedding_node),
    ("llm_relation", llm_relation_node),
]

def run_multi_table_pipeline(state: MultiTableGraphState, config: Optional[Dict[str, Any]] = None) -> MultiTableGraphState:
    from Tabular_to_Neo4j.utils.output_saver import get_output_saver
    from Tabular_to_Neo4j.utils.prompt_utils import save_prompt_sample
    import logging
    logger = logging.getLogger(__name__)
    output_saver = get_output_saver()
    # Per-table phase (use the StateGraph abstraction for each table)
    graph_template = create_multi_table_graph()
    from Tabular_to_Neo4j.utils.state_saver import save_state_snapshot
    # For each table, run the compiled graph node-by-node, saving state and prompts after each node
    for table_name, table_state in state.items():
        logger.info(f'[PER_TABLE][{table_name}][BEFORE_PIPELINE] Initial state type: {type(table_state).__name__}')
        compiled_graph = graph_template.compile()
        # Get the pipeline node order
        node_order_map = {name: idx+1 for idx, (name, _) in enumerate(PIPELINE_NODES)}
        # Manually step through nodes
        current_state = table_state
        import inspect
        for node_idx, (node_name, node_func) in enumerate(PIPELINE_NODES, 1):
            try:
                prev_state = current_state.copy() if hasattr(current_state, 'copy') else dict(current_state)
                # Inspect function signature to determine if 'config' should be passed
                sig = inspect.signature(node_func)
                if 'config' in sig.parameters:
                    current_state = node_func(current_state, config)
                else:
                    current_state = node_func(current_state)
                # Save node output
                if output_saver:
                    output_saver.save_node_output(node_name, current_state, node_order=node_idx, table_name=table_name)
                # Save state snapshot
                save_state_snapshot({table_name: current_state}, timestamp=output_saver.timestamp, base_dir=output_saver.base_dir)
                # Save prompt (if applicable)
                try:
                    save_prompt_sample(
                        template_name=f"{node_name}.txt",
                        formatted_prompt="",  # If you have the prompt, pass it here
                        kwargs={"state_name": node_name, "table_name": table_name},
                        base_dir=output_saver.base_dir,
                        timestamp=output_saver.timestamp,
                        table_name=table_name,
                        subfolder="prompts"
                    )
                except Exception as prompt_exc:
                    logger.debug(f"Prompt sample not saved for node '{node_name}': {prompt_exc}")
                # Log type and truncated value after node execution
                short_val = str(current_state)
                if len(short_val) > 300:
                    short_val = short_val[:300] + '...'
                logger.info(f'[PER_TABLE][{table_name}][NODE_EXEC][AFTER][{node_name}] Type: {type(current_state).__name__}, Value: {short_val}')
            except Exception as e:
                logger.error(f'[PER_TABLE][{table_name}][NODE_EXEC][{node_name}] Exception: {e}, State type: {type(current_state).__name__}, State value: {current_state}')
                raise

        # Update state for this table
        state[table_name] = current_state
        logger.info(f'[PER_TABLE][{table_name}][AFTER_PIPELINE] State type: {type(current_state).__name__}')

    # Diagnostic logging: print types after per-table pipelines, before cross-table nodes
    logger.info('[PER_TABLE][AFTER] Table states: ' + str({k: type(v).__name__ for k, v in state.items()}))
    # Enforce that all table states are valid mapping types (GraphState, AddableValuesDict, or MutableMapping)
    from collections.abc import MutableMapping
    try:
        from langchain_core.utils import AddableValuesDict
    except ImportError:
        AddableValuesDict = None
    for table_name, table_state in state.items():
        if not (
            isinstance(table_state, (GraphState, MutableMapping)) or
            (AddableValuesDict and isinstance(table_state, AddableValuesDict))
        ):
            logger.error(f"[PER_TABLE][{table_name}] Invalid state type after pipeline: {type(table_state).__name__}")
            raise TypeError(f"Table state for '{table_name}' must be a GraphState, AddableValuesDict, or MutableMapping, got {type(table_state)}")
    # Cross-table phase
    from Tabular_to_Neo4j.utils.state_saver import save_state_snapshot
    import inspect
    for node_idx, (node_name, node_func) in enumerate(CROSS_TABLE_NODES, 1):
        logger.info(f'[CROSS_TABLE][{node_name}][BEFORE] Table states: ' + str({k: type(v).__name__ for k, v in state.items()}))
        sig = inspect.signature(node_func)
        if 'config' in sig.parameters:
            state = node_func(state, config)
        else:
            state = node_func(state)
        # Save cross-table node output
        if output_saver:
            output_saver.save_node_output(node_name, state, node_order=node_idx, table_name="inter_table")
        # Save state snapshot after each cross-table node
        save_state_snapshot({"inter_table": state}, timestamp=output_saver.timestamp, base_dir=output_saver.base_dir)
        # Save prompt (if applicable)
        try:
            save_prompt_sample(
                template_name=f"{node_name}.txt",
                formatted_prompt="",  # If you have the prompt, pass it here
                kwargs={"state_name": node_name, "table_name": "inter_table"},
                base_dir=output_saver.base_dir,
                timestamp=output_saver.timestamp,
                table_name="inter_table",
                subfolder="prompts"
            )
        except Exception as prompt_exc:
            logger.debug(f"Prompt sample not saved for cross-table node '{node_name}': {prompt_exc}")
        # Log type and truncated value for each table after cross-table node
        for k, v in state.items():
            short_val = str(v)
            if len(short_val) > 300:
                short_val = short_val[:300] + '...'
            logger.info(f'[CROSS_TABLE][{node_name}][NODE_EXEC][AFTER][{k}] Type: {type(v).__name__}, Value: {short_val}')
        logger.info(f'[CROSS_TABLE][{node_name}][AFTER] Table states: ' + str({k: type(v).__name__ for k, v in state.items()}))
    # Ensure all table states are valid mapping types (GraphState, AddableValuesDict, or MutableMapping) after cross-table nodes
    from collections.abc import MutableMapping
    try:
        from langchain_core.utils import AddableValuesDict
    except ImportError:
        AddableValuesDict = None
    for table_name, table_state in state.items():
        if not (
            isinstance(table_state, (GraphState, MutableMapping)) or
            (AddableValuesDict and isinstance(table_state, AddableValuesDict))
        ):
            logger.error(f"[FINAL][{table_name}] Invalid state type after cross-table nodes: {type(table_state).__name__}")
            raise TypeError(f"Table state for '{table_name}' must be a GraphState, AddableValuesDict, or MutableMapping after cross-table nodes, got {type(table_state)}")
    return state
    
def initialize_multi_table_state(table_folder: str) -> MultiTableGraphState:
    """
    Initializes a MultiTableGraphState for all tables in the folder.
    For each CSV, also attempts to find and store the corresponding metadata file path (as in the one-table case).
    Each key is a table name, and the value is a full GraphState instance.
    """
    state = MultiTableGraphState()
    for fname in os.listdir(table_folder):
        if fname.lower().endswith('.csv'):
            table_name = os.path.splitext(fname)[0]
            csv_path = os.path.join(table_folder, fname)
            metadata_path = get_metadata_path_for_csv(csv_path)
            state[table_name] = GraphState(csv_file_path=csv_path, metadata_file_path=metadata_path)
    return state
