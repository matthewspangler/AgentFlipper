# Flipper Zero AI Agent

A Python solution that connects your Flipper Zero to LiteLLM with Ollama as the backend, allowing you to control your Flipper Zero using natural language commands with context-aware conversations.

## Overview

- Controls Flipper Zero over serial via PyFlipper
- Connects to Ollama or whatever LLM provider you prefer, with LiteLLM
- Uses Retrieval-Augmented Generation (RAG) make sure the LLM is aware of Flipper documentation
- Maintains conversation context for natural follow-up commands
- Intelligently executes multi-step tasks
- Comprehensive logging for debugging and analysis

For technical details, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Requirements

- Python 3.9+ and pip dependencies listed in requirements.txt
- Flipper Zero device
- Ollama or an LLM provider

## Installation

### Using Setup Script

```bash
./setup.sh
```

### Manual

1. Clone this repository with submodules:

```bash
git clone --recursive https://github.com/yourusername/flipper-agent
cd flipper-agent
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Pull the recommended Qwen model in Ollama:

```bash
ollama pull qwen2.5-coder:14b
```

4. Fill out config.yaml.example and rename to config.yaml

## Usage

1. First, build the documentation database (only needed once):

```bash
./flipper_docs_loader.py
```

2. Connect your Flipper Zero to your computer via USB

3. Run the agent:

```bash
./run.sh
```

Or run the Python script directly:

```bash
./flipper_agent_with_rag.py
```

4. Enter your commands in natural language:
   - "Show device information and status"
   - "Turn on the green LED and set backlight to 50%"
   - "Scan for NFC tags and save any found ones"
   - "Help me clone this NFC tag to file 'mycard'"
   - "Analyze the last SubGHz scan results"

The RAG system will retrieve relevant CLI documentation to help generate more accurate commands.

## Command-line Options

The agent supports these options:

- `--port PORT`: Specify the serial port (default: /dev/ttyACM0)
- `--config FILE`: Use a custom config file
- `--model MODEL`: Use a specific model (e.g., "ollama/qwen2.5-coder:14b")
- `--no-rag`: Disable RAG retrieval system
- `--max-history-tokens TOKENS`: Set maximum token count for conversation history
- `--console-log`: Show log messages in console (default: log to file only)
- `--console-log`: Show log messages in console (default: log to file only)
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging verbosity

## Configuration

The config file is loaded from:
1. `~/.config/flipper_agent/config.yaml` (if exists)
2. `./config.yaml` (project directory)

## Licensing

GNU General Public License v3.0 (GPLv3)