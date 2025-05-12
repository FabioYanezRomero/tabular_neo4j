#!/usr/bin/env python3
"""
Test script to demonstrate the Tabular to Neo4j converter with a sample CSV file.
"""

import os
import sys
from Tabular_to_Neo4j.utils.logging_config import get_logger
from pathlib import Path

# Add the parent directory to the path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from Tabular_to_Neo4j.main import run_analysis

# Set up logging

# Configure logging
logger = get_logger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def main():
    """
    Run the analysis on the sample CSV file.
    """
    # Get the path to the sample CSV file
    sample_file = os.path.join(os.path.dirname(__file__), 'sample_data', 'customers.csv')
    
    # Run the analysis
    logger.info(f"Running analysis on sample file: {sample_file}")
    result = run_analysis(sample_file, verbose=True)
    
    # The results are already printed by the run_analysis function when verbose=True
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
