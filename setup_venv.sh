#!/bin/bash

# Exit on error
set -e

echo "Setting up clean virtual environment..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models logs

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "Please update .env with your actual API keys and configuration"
fi

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate" 