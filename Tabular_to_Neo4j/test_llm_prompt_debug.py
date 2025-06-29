from Tabular_to_Neo4j.utils.prompt_utils import format_prompt
from Tabular_to_Neo4j.utils.output_saver import initialize_output_saver

def dummy_llm_node():
    # Ensure OutputSaver is initialized
    initialize_output_saver()
    # Simulate a minimal state as used in header_inference
    state = {
        "table_name": "dummy_table",
        "csv_file_path": "dummy.csv"
    }
    # Minimal variables to fill the template for infer_header.txt
    variables = {
        "metadata_text": "Sample metadata for debugging.",
        "num_columns": 3,
        "data_sample": "1,2,3\n4,5,6"
    }
    table_name = state.get("table_name")
    prompt = format_prompt("infer_header.txt", table_name=table_name, **variables)
    print("--- Loaded and Formatted Prompt ---")
    print(prompt)
    print("--- END ---")

if __name__ == "__main__":
    dummy_llm_node()
