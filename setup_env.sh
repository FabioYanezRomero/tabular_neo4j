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
    echo "Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        cat > .env << EOF
# LLM API Configuration
LLM_API_KEY=""
LLM_PROVIDER="lmstudio"
LMSTUDIO_HOST="127.0.0.1"
LMSTUDIO_PORT="1234"
LMSTUDIO_BASE_URL="http://127.0.0.1:1234/"
EOF
    fi
    echo ".env file created. Please edit it to add your API keys and settings."
fi

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""

echo "Note: Make sure to configure your LLM provider in the .env file or in Tabular_to_Neo4j/config.py"
