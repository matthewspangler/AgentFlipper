#!/bin/bash

# AgentFlipper Run Script
# This script activates the virtual environment, runs the doc downloader if needed,
# and then runs the Flipper Zero AI Agent

set -e

# Define colors for output
GREEN='\033[0;32m'
PURPLE='\033[0;95m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse command line arguments
SKIP_DOCS=false
PORT=""
MODEL=""
NO_RAG=false
ENHANCED_PROMPT=false
MAX_HISTORY_TOKENS=""
LOG_LEVEL="INFO"
CONSOLE_LOG=false

# Display usage information
function show_usage {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  -h, --help            Show this help message and exit"
    echo -e "  -s, --skip-docs       Skip documentation download/update"
    echo -e "  -e, --enhanced-prompt Use enhanced system prompt with downloaded CLI documentation"
    echo -e "  -p, --port PORT       Specify serial port for Flipper Zero"
    echo -e "  -m, --model MODEL     Specify LLM model to use"
    echo -e "  -t, --tokens TOKENS   Maximum token count for conversation history (default: 2000)"
    echo -e "  -l, --log LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    echo -e "  -c, --console-log     Show log messages in console (default: log to file only)"
    echo -e "  --no-rag              Run RAG agent but disable RAG"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_usage
            ;;
        -s|--skip-docs)
            SKIP_DOCS=true
            shift
            ;;
        -e|--enhanced-prompt)
            ENHANCED_PROMPT=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -m|--model)
            MODEL="$2"
            shift
            shift
            ;;
        --no-rag)
            NO_RAG=true
            shift
            ;;
        -t|--tokens)
            MAX_HISTORY_TOKENS="$2"
            shift
            shift
            ;;
        -l|--log)
            LOG_LEVEL="$2"
            shift
            shift
            ;;
        -c|--console-log)
            CONSOLE_LOG=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $key${NC}"
            show_usage
            ;;
    esac
done

echo -e "\033[1;37m======================================\033[0m"
echo -e "\033[1;37m  Flipper Zero AI Agent - Run Script  \033[0m"
echo -e "\033[1;37m======================================\033[0m"

# Check if virtual environment exists
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "\n${GREEN}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"


# Check if documentation has been downloaded, and download if needed
# Only check if RAG is NOT explicitly disabled via --no-rag passed to main.py
# and download is NOT skipped via script options (--skip-docs, --no-rag-download).
DOCS_DIR="$SCRIPT_DIR/docs"
VECTOR_STORE_PATH="$DOCS_DIR/flipper_cli_faiss" # Assuming this is the path used by docs_loader.py

# Check if --no-rag is among the arguments passed to the script *before* passing them to main.py
# This allows the script to potentially skip the doc download if RAG is disabled in main.py.
NO_RAG_FLAG_PRESENT_FOR_MAIN=false
for arg in "$@"; do
    if [ "$arg" == "--no-rag" ]; then
        NO_RAG_FLAG_PRESENT_FOR_MAIN=true
        break
    fi
done


if [ "$SKIP_DOCS" = false ] && [ "$NO_RAG_FLAG_PRESENT_FOR_MAIN" = false ]; then
     if [ ! -d "$VECTOR_STORE_PATH" ]; then
        echo -e "\n${GREEN}Documentation vector store not found. Downloading and processing documentation...${NC}"
        # Need to run docs_loader.py with python from the activated venv
        python "$SCRIPT_DIR/docs_loader.py"
    else
        echo -e "\n${GREEN}Documentation vector store found. Skipping download check.${NC}"
        echo -e "${GREEN}(Use -s, --skip-docs, or --no-rag-download script options to skip this check)${NC}"
    fi
elif [ "$SKIP_DOCS" = true ]; then
    echo -e "\n${YELLOW}Documentation download check skipped by script options.${NC}"
elif [ "$NO_RAG_FLAG_PRESENT_FOR_MAIN" = true ]; then
    echo -e "\n${YELLOW}--no-rag flag detected in arguments for main.py. Skipping documentation download check.${NC}"
fi


# Run the agent
echo -e "\n${PURPLE}Starting AgentFlipper...${NC}"
# Execute main.py directly
# Pass all remaining arguments ($@) directly to the main.py script
python main.py "$@" > "$SCRIPT_DIR/logs/agent_flipper_tui_$(date +'%Y%m%d_%H%M%S').log"

# The script finishes here after main.py exits