from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd

class GraphState(TypedDict):
    csv_file_path: str
    raw_dataframe: Optional[pd.DataFrame]
    has_header_heuristic: Optional[bool]  # Result of initial heuristic check
    header_row_if_present: Optional[List[str]]  # Stores the original first row if heuristic says header
    
    # Header processing state
    inferred_header: Optional[List[str]]
    validated_header: Optional[List[str]]
    is_header_correct_llm: Optional[bool]
    header_correction_suggestions: Optional[str]
    translated_header: Optional[List[str]]
    is_header_in_target_language: Optional[bool]
    final_header: Optional[List[str]]  # The header to be used for the DataFrame
    processed_dataframe: Optional[pd.DataFrame]  # DataFrame with the final_header applied

    # Column analysis state
    column_analytics: Optional[Dict[str, Dict[str, Any]]]  # {col_name: {uniqueness: 0.9, ...}}
    llm_column_semantics: Optional[Dict[str, Dict[str, Any]]]  # {col_name: {semantic_type: "City", neo4j_role: "POTENTIAL_LINKED_NODE", ...}}

    # Final output
    inferred_neo4j_schema: Optional[Dict[str, Any]]  # { "primary_entity_label": "inferred_from_filename", "columns": [{ "name": "col_A", "role": "NODE_PROPERTY", "neo4j_property": "propA"}, ...]}
    error_messages: List[str]
