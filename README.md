# Flipper Zero AI Agent

A Python solution that connects your Flipper Zero to LiteLLM with Ollama as the backend, allowing you to control your Flipper Zero using natural language commands with context-aware conversations.

## Overview

This project uses a RAG-enhanced agent that offers:

- Direct communication with Flipper Zero via PyFlipper
- Connects to Ollama for local LLM processing
- Uses Retrieval-Augmented Generation (RAG) for better command quality
- Provides relevant Flipper Zero CLI documentation to the LLM
- Maintains conversation context for natural follow-up commands
- Intelligently executes multi-command sequences
- Processes multi-step tasks with automatic follow-up actions
- Comprehensive logging for debugging and analysis

For technical details about the agent architecture, special markers system, and token-efficient context management, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Requirements

- Python 3.6+
- Flipper Zero device
- Ollama running locally
- PyFlipper (included as a submodule)
- LangChain and related libraries

## Installation

1. Clone this repository with submodules:

```bash
git clone --recursive https://github.com/yourusername/flipper-agent
cd flipper-agent
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Pull the Qwen model in Ollama (or another model of your choice):

```bash
ollama pull qwen2.5-coder:14b
```

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
   - "Show me device information"
   - "Turn on the green LED"
   - "Scan for NFC tags"
   - "Display the current date"

The RAG system will retrieve relevant CLI documentation to help generate more accurate commands.

## Command-line Options

The agent supports these options:

- `--port PORT`: Specify the serial port (default: /dev/ttyACM0)
- `--config FILE`: Use a custom config file
- `--model MODEL`: Use a specific model (e.g., "llama3:latest")
- `--no-rag`: Disable RAG retrieval system
- `--max-history-tokens TOKENS`: Set maximum token count for conversation history
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging verbosity
- `--console-log`: Show log messages in console (default: log to file only)
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging verbosity

You can also use the `run.sh` script which provides a more user-friendly interface:

```bash
./run.sh --help               # Show all options
./run.sh -p /dev/ttyACM1      # Use a different port
./run.sh -m llama3:latest     # Use a different model
./run.sh -t 3000              # Increase token history limit
./run.sh --skip-docs          # Skip documentation check
```

## Configuration

The config file is loaded from:
1. `~/.config/flipper_agent/config.yaml` (if exists)
2. `./config.yaml` (project directory)

Example config:
```yaml
flipper:
  port: /dev/ttyACM0
  timeout: 5.0
llm:
  provider: ollama
  model: qwen2.5-coder:14b
  api_base: http://localhost:11434
  max_history_tokens: 2000
  user_prompt: >
    You are an assistant specialized in controlling the Flipper Zero device.
    Try to be efficient with your commands, and provide accurate results.
```

## Prompt System

The agent uses a two-prompt approach:

1. **System Prompt**: Built into the code with special markers and format instructions
   - Controls command formatting and special markers
   - Ensures consistent output formatting
   - Not user-configurable (for stability)

2. **User Prompt**: Configurable in config.yaml
   - Customize the assistant's personality and focus
   - Add domain-specific instructions
   - Won't interfere with core system functionality

## How It Works

1. **PyFlipper Integration**:
   - Communicates with Flipper Zero via serial connection
   - Sends commands and receives responses

2. **LangChain RAG System**:
   - Processes Flipper Zero CLI documentation
   - Creates vector embeddings for efficient retrieval
   - Retrieves relevant documentation based on user queries

3. **LiteLLM Integration**:
   - Connects to Ollama to access powerful LLMs
   - Enhances prompts with retrieved documentation
   - Maintains conversation context between commands
   - Intelligently manages token-based history
   - Generates accurate CLI commands

## Advanced Features

### Context-Aware Conversations

The agent maintains conversation context between interactions, enabling more natural follow-up commands:

1. **Follow-up Commands**: The agent remembers previous commands and their context
   ```
   > Turn on the backlight
   $ backlight on
   
   > Set it to 50%
   $ backlight 50
   ```

2. **Token-Based History**: Conversation history is managed using a token-based approach (industry standard)
   - Configure in `config.yaml` with the `max_history_tokens` parameter
   - Older conversation exchanges are removed when the token limit is reached

3. **Command vs. Information Requests**:
   - By default, the agent assumes you want to execute a command on the Flipper Zero
   - For information requests, the agent will respond with helpful text prefixed with "INFO:"
   - Example: "How does NFC work?" → "INFO: NFC (Near Field Communication)..."

### Multi-Command Execution

The agent can now parse and execute multiple commands from a single user request:

1. **Command Sequences**: Multiple commands can be specified in a single prompt
   ```
   > Turn on the green LED and set the backlight to 50%
   
   Executing 2 commands:
   1. led g on
   2. backlight 50
   
   $ led g on
   Response: ...
   
   $ backlight 50
   Response: ...
   ```

2. **Command Visualization**: All commands are displayed in orange text for easy identification

### Task Tracking

The agent now tracks tasks from start to completion:

1. **Multi-Step Operations**: For complex operations requiring multiple steps, the agent will:
   - Execute initial commands
   - Analyze the results
   - Automatically generate follow-up commands based on the output
   - Continue until the task is completed

2. **Task Completion**: The agent intelligently detects when a task is completed

### Comprehensive Logging

The agent includes robust logging capabilities:

1. **Log Files**: All operations are logged to timestamped files in the `logs` directory
2. **Configurable Log Level**: Set logging verbosity with `--log-level` parameter
3. **Command & Response Tracking**: All commands and responses are logged for debugging

## Troubleshooting

1. **Connection Issues**:
   - Check the correct port in configuration
   - Make sure PyFlipper submodule is initialized
   - Ensure the Flipper Zero is connected via USB

2. **LLM Issues**:
   - Verify Ollama is running (`ps aux | grep ollama`)
   - Check that the model is downloaded (`ollama list`)
   - Try with the `--model` parameter to specify a different model

3. **RAG Issues**:
   - Make sure you've run the documentation loader first
   - Check the docs directory exists with vector store files
   - Try running with `--no-rag` to bypass RAG functionality

4. **Context Issues**:
   - If the agent seems to forget context, try increasing the history token limit
   - Use more specific references when the conversation becomes complex
   - If needed, specify context explicitly in your commands

## License

MIT