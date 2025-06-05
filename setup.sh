#!/bin/bash

# Flipper Zero AI Agent Setup Script
# This script creates a Python virtual environment and installs all dependencies

set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Flipper Zero AI Agent - Setup Script  ${NC}"
echo -e "${GREEN}======================================${NC}"

# Get directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is installed
echo -e "\n${YELLOW}Checking for Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3 and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}Python found: $(python3 --version)${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Do you want to recreate it? [y/N]:${NC}"
    read -r recreate
    if [[ "$recreate" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${GREEN}Using existing virtual environment.${NC}"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
fi

# Activate the virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if PyFlipper submodule is initialized
echo -e "\n${YELLOW}Checking PyFlipper submodule...${NC}"
if [ ! -d "$SCRIPT_DIR/pyFlipper/src/pyflipper" ]; then
    echo -e "${YELLOW}PyFlipper submodule not initialized. Initializing...${NC}"
    if [ ! -d "$SCRIPT_DIR/.git" ]; then
        echo -e "${YELLOW}Initializing git repository...${NC}"
        git init
    fi
    
    if [ -d "$SCRIPT_DIR/pyFlipper" ]; then
        echo -e "${YELLOW}Removing existing pyFlipper directory...${NC}"
        rm -rf "$SCRIPT_DIR/pyFlipper"
    fi
    
    echo -e "${YELLOW}Adding PyFlipper as a submodule...${NC}"
    git submodule add https://github.com/wh00hw/pyFlipper.git
    
    echo -e "${YELLOW}Updating submodules...${NC}"
    git submodule update --init --recursive
else
    echo -e "${GREEN}PyFlipper submodule already initialized.${NC}"
fi

# Install PyFlipper in development mode
echo -e "\n${YELLOW}Installing PyFlipper in development mode...${NC}"
pip install -e ./pyFlipper

# Make scripts executable
echo -e "\n${YELLOW}Making scripts executable...${NC}"
chmod +x flipper_agent_with_rag.py
chmod +x flipper_docs_loader.py
chmod +x run.sh

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}  Setup completed successfully!  ${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${YELLOW}To run the Flipper Zero AI Agent, use:${NC}"
echo -e "${GREEN}  ./run.sh${NC}"
echo -e "${YELLOW}Or activate the virtual environment manually:${NC}"
echo -e "${GREEN}  source venv/bin/activate${NC}"
echo -e "${GREEN}  ./flipper_agent_with_rag.py${NC}"
echo -e "${GREEN}======================================${NC}"