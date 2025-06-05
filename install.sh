#!/bin/bash

# Flipper Zero AI Agent - Simple Installer
# Installs dependencies without creating config files

set -e

echo "======================================"
echo "Flipper Zero AI Agent - Installer"
echo "======================================"

# Check if Python is installed
echo "Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3 and try again."
    exit 1
fi

echo "Python found: $(python3 --version)"

# Install required Python packages
echo -e "\nInstalling required Python packages..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
else
    python3 -m pip install litellm pyserial pyyaml
fi

# Make script executable
echo -e "\nMaking flipper_agent.py executable..."
chmod +x flipper_agent.py

echo -e "\n======================================"
echo "Installation completed successfully!"
echo ""
echo "To configure the agent:"
echo "  mkdir -p ~/.config/flipper_agent"
echo "  cp config.yaml ~/.config/flipper_agent/"
echo ""
echo "To run the Flipper Zero AI Agent:"
echo "  ./flipper_agent.py"
echo ""
echo "For more information and examples, see:"
echo "  README.md and examples.md"
echo "======================================"