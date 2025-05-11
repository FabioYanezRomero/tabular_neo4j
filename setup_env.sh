#!/bin/bash
# Setup script for Tabular_to_Neo4j project

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create a .env file for configuration
if [ ! -f .env ]; then
    echo "Creating .env file for configuration..."
    cat > .env << EOF
# LLM API Configuration
LLM_API_KEY=""  # Set your API key here
LLM_PROVIDER="lmstudio"  # Options: "openai", "ollama", "anthropic", "lmstudio", "huggingface"
LMSTUDIO_BASE_URL="http://localhost:1234/v1"  # Default LMStudio local server
EOF
    echo ".env file created. Please edit it to add your API keys."
fi

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To test the system with the sample data, run:"
echo "python Tabular_to_Neo4j/test_sample.py"
echo ""
echo "Note: Make sure to configure your LLM provider in the .env file or in Tabular_to_Neo4j/config.py"
