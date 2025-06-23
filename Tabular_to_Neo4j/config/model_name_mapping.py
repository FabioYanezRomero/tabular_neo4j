"""
Mapping from LMStudio model IDs to their Ollama (quantized) equivalents.
Add or adjust mappings as new quantized models become available.
"""

MODEL_NAME_MAPPING = {
    # LMStudio model : Ollama QAT model (string)
    "gemma-3-1b-it": "gemma3:1b-it-qat",
    "gemma-3-4b-it": "gemma3:4b-it-qat",
    "gemma-3-12b-it": "gemma3:12b-it-qat",
    "gemma-3-27b-it": "gemma3:27b-it-qat",
    # Other models (keep as-is or update if QAT equivalents are known)
    "qwen3-8b": "qwen3:8b-instruct-q4_K_M",
    "qwen3-14b": "qwen3:14b-instruct-q4_K_M",
    "qwen3-4b": "qwen3:4b-instruct-q4_K_M",
    "text-embedding-nomic-embed-text-v1.5": "nomic-embed-text:v1.5-q4_K_M",
    "salamandra-7b-instruct-aina-hack": "salamandra:7b-instruct-q4_K_M",
    "deepseek-r1-distill-llama-8b": "deepseek-llama:8b-distill-q4_K_M",
    "deepseek-r1-distill-qwen-7b": "deepseek-qwen:7b-distill-q4_K_M",
    # Add more mappings as needed
}

