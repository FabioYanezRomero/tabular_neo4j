from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os

LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

def get_lm_studio_model():
    return ChatOpenAI(
        base_url=LMSTUDIO_URL,
        api_key="lm-studio"
    )

def get_ollama_model(model_name=None):
    from langchain_community.chat_models import ChatOllama
    if model_name is None:
        model_name = OLLAMA_MODEL
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=model_name
    )

def list_ollama_models():
    import requests
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        return []

# For backward compatibility
models = {
    "lm_studio": get_lm_studio_model,
    "ollama": get_ollama_model,
}

