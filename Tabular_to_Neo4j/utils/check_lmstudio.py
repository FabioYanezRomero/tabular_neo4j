#!/usr/bin/env python3
"""
Utility script to check if LMStudio server is reachable.
"""

import argparse
import sys
import requests
import time

def check_lmstudio_connection(host="host.docker.internal", port=1234, retries=3, retry_delay=2):
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
            print(f"Attempt {attempt + 1}/{retries}: Connecting to LMStudio at {url}...")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Connection successful! LMStudio server is reachable.")
                print(f"Available models: {response.json()}")
                return True
            else:
                print(f"❌ Connection failed with status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            
        if attempt < retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print("❌ All connection attempts failed. Please check if LMStudio is running and properly configured.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Check if LMStudio server is reachable")
    parser.add_argument("--host", default="host.docker.internal", help="LMStudio host address")
    parser.add_argument("--port", type=int, default=1234, help="LMStudio port number")
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
