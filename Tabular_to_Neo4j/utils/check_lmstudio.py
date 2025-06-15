#!/usr/bin/env python3
"""
Utility script to check if LMStudio server is reachable.
"""

import argparse
import sys
import requests
import time
import os

from Tabular_to_Neo4j.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

def check_lmstudio_connection(host=None, port=None, retries=3, retry_delay=2):
    # Use environment variables if host or port are not provided
    if host is None:
        host = os.environ.get("LMSTUDIO_HOST", "host.docker.internal")
    if port is None:
        port = os.environ.get("LMSTUDIO_PORT", 1234)
    """
    Check if LMStudio server is reachable.
    
    Args:
        host: Host address (default: host.docker.internal for Docker)
        port: Port number (default: 1234)
        retries: Number of connection retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    url = f"http://{host}:{port}/v1/models"

    for attempt in range(retries):
        try:
            logger.info(
                f"Attempt {attempt + 1}/{retries}: Connecting to LMStudio at {url}..."
            )
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                logger.info("✅ Connection successful! LMStudio server is reachable.")
                logger.debug("Available models: %s", response.json())
                return True
            else:
                logger.error(
                    "❌ Connection failed with status code: %s", response.status_code
                )

        except requests.exceptions.RequestException as e:
            logger.error("❌ Connection error: %s", e)

        if attempt < retries - 1:
            logger.info("Retrying in %s seconds...", retry_delay)
            time.sleep(retry_delay)

    logger.error(
        "❌ All connection attempts failed. Please check if LMStudio is running and properly configured."
    )
    return False

def main():
    setup_logging()

    # Get default values from environment variables
    default_host = os.environ.get("LMSTUDIO_HOST", "host.docker.internal")
    default_port = int(os.environ.get("LMSTUDIO_PORT", 1234))
    
    parser = argparse.ArgumentParser(description="Check if LMStudio server is reachable")
    parser.add_argument("--host", default=default_host, help=f"LMStudio host address (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"LMStudio port number (default: {default_port})")
    
    parser.add_argument("--retries", type=int, default=3, help="Number of connection retries")
    parser.add_argument("--retry-delay", type=int, default=2, help="Delay between retries in seconds")
    
    args = parser.parse_args()
    
    success = check_lmstudio_connection(
        host=args.host,
        port=args.port,
        retries=args.retries,
        retry_delay=args.retry_delay
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
