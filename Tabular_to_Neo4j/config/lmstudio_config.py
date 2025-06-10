"""Configuration for LMStudio integration."""

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file if present
load_dotenv(find_dotenv())

# LMStudio server configuration
LMSTUDIO_HOST = os.getenv("LMSTUDIO_HOST", "127.0.0.1")
LMSTUDIO_PORT = int(os.getenv("LMSTUDIO_PORT", 1234))
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}")
LMSTUDIO_API_VERSION = os.getenv("LMSTUDIO_API_VERSION", "v1")
LMSTUDIO_ENDPOINT = os.getenv("LMSTUDIO_ENDPOINT", f"{LMSTUDIO_BASE_URL}/{LMSTUDIO_API_VERSION}")

# Default model to use
DEFAULT_MODEL = os.getenv("LMSTUDIO_DEFAULT_MODEL", "local-model")

# API parameters
DEFAULT_TEMPERATURE = float(os.getenv("LMSTUDIO_DEFAULT_TEMPERATURE", 0.0))
DEFAULT_MAX_TOKENS = int(os.getenv("LMSTUDIO_DEFAULT_MAX_TOKENS", 1024))
DEFAULT_TOP_P = float(os.getenv("LMSTUDIO_DEFAULT_TOP_P", 0.95))
