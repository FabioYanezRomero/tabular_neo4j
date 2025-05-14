"""
Configuration for LMStudio integration.
"""

# LMStudio server configuration
LMSTUDIO_HOST = "host.docker.internal"  # Special Docker DNS name that resolves to the host machine
LMSTUDIO_PORT = 1234
LMSTUDIO_BASE_URL = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"
LMSTUDIO_API_VERSION = "v1"
LMSTUDIO_ENDPOINT = f"{LMSTUDIO_BASE_URL}/{LMSTUDIO_API_VERSION}"

# Default model to use
DEFAULT_MODEL = "local-model"  # This will be replaced with the actual model name from LMStudio

# API parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 0.95
