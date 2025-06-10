"""
Configuration for LMStudio integration.
"""

# LMStudio server configuration
import os

LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "127.0.0.1")
LMSTUDIO_PORT = int(os.environ.get("LMSTUDIO_PORT", 1234))
LMSTUDIO_API_VERSION = os.environ.get("LMSTUDIO_API_VERSION", "v1")
LMSTUDIO_BASE_URL = os.environ.get(
    "LMSTUDIO_BASE_URL", f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"
)
LMSTUDIO_ENDPOINT = f"{LMSTUDIO_BASE_URL}/{LMSTUDIO_API_VERSION}"

# Default model to use
DEFAULT_MODEL = os.environ.get("LMSTUDIO_DEFAULT_MODEL", "local-model")

# API parameters
DEFAULT_TEMPERATURE = float(os.environ.get("LMSTUDIO_DEFAULT_TEMPERATURE", 0.0))
DEFAULT_MAX_TOKENS = int(os.environ.get("LMSTUDIO_DEFAULT_MAX_TOKENS", 1024))
DEFAULT_TOP_P = float(os.environ.get("LMSTUDIO_DEFAULT_TOP_P", 0.95))
