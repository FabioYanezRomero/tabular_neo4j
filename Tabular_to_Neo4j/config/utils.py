"""
Mapping from LMStudio model IDs to their Ollama (quantized) equivalents.
Add or adjust mappings as new quantized models become available.
"""

MODEL_NAME_MAPPING = {
    "text-embedding-nomic-embed-text-v1.5": "nomic-embed-text",    # Balanced, default
    "bge-base-en-v1.5": "mxbai-embed-large",                       # High accuracy, larger model
    "all-MiniLM-L6-v2": "all-minilm", 
    # LMStudio model : Ollama QAT model (string)
    "gemma-3-1b-it": "gemma3:1b-it-qat",
    "gemma-3-4b-it": "gemma3:4b-it-qat",
    "gemma-3-12b-it": "gemma3:12b-it-qat",
    "gemma-3-27b-it": "gemma3:27b-it-qat",
    # Other models (keep as-is or update if QAT equivalents are known)
    "qwen3-8b": "qwen3:8b-instruct-q4_K_M",
    "qwen3-14b": "qwen3:14b-q4_K_M",
    "qwen3-4b": "qwen3:4b-instruct-q4_K_M",
    "text-embedding-nomic-embed-text-v1.5": "nomic-embed-text:v1.5-q4_K_M",
    "salamandra-7b-instruct-aina-hack": "salamandra:7b-instruct-q4_K_M",
    "deepseek-r1-distill-llama-8b": "deepseek-llama:8b-distill-q4_K_M",
    "deepseek-r1-distill-qwen-7b": "deepseek-qwen:7b-distill-q4_K_M",
    # Add more mappings as needed
}


# Per-State LLM Configuration with GGUF models through LMStudio
# Each state has its own model and output format specification
class DynamicLLMConfigs(dict):
    """
    A dictionary that returns the config for 'infer_entity_relationships' for any key starting with 'infer_relationship_'.
    """
    def __getitem__(self, key):
        if key in super().keys():
            return super().__getitem__(key)
        if key.startswith('infer_relationship_'):
            return super().__getitem__('infer_entity_relationship_pair')
        raise KeyError(key)

    def get(self, key, default=None):
        if key in super().keys():
            return super().get(key, default)
        if key.startswith('infer_relationship_'):
            return super().get('infer_entity_relationship_pair', default)
        return default