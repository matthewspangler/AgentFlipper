# Flipper Zero AI Agent Architecture

This document describes the architecture and technical details of the Flipper Zero AI Agent.

## Core Components

### 1. PyFlipper Integration
- Direct communication with Flipper Zero via serial connection
- Command sending and response parsing
- Device state management

### 2. LLM Integration
- Connection to LiteLLM API (currently configured for Ollama)
- Prompt engineering and response parsing
- Context management and history tracking

### 3. RAG System
- Vector database of Flipper Zero CLI documentation
- Semantic search for relevant commands
- Context enhancement for more accurate command generation

## Special Markers System

The agent uses special markers for structured communication between components:

### 1. Command Marker
The `:COMMAND:` marker explicitly identifies content that should be executed as commands.

```
:COMMAND:
led g 255
led bl 255
```

This marker ensures that:
- Only intended commands are sent to the Flipper Zero
- Other text/markers won't interfere with command execution
- Command boundaries are clearly delineated

### 2. Task Completion
The `:TASK_COMPLETE:` marker indicates a task has been fully finished.

```
:COMMANDS:
led g 255
:SUMMARY: Green LED turned on
:TASK_COMPLETE:
```

When the LLM includes this marker, the agent will:
- Mark the current task as complete
- Display completion message to the user
- Reset task tracking state
- Return control to the user

### 3. Information Responses
The `:INFO: [text]` marker is used for non-command informational content.

```
:INFO: The Flipper Zero has RGB LEDs that can display different colors
```

When detected, the agent will:
- Display the information to the user
- Skip command execution
- Store the information in conversation history

## Token-Efficient Context Management

The agent uses advanced techniques to manage the token context window efficiently:

### Progressive Summarization
- Command sequences are summarized to preserve context while using fewer tokens
- Summaries replace verbose exchanges in the conversation history
- This enables longer conversations while staying within token limits

### Smart History Pruning
- Token count is estimated and tracked for all history entries
- When token limit is reached, oldest entries are pruned
- Pruning preserves essential context when possible

### History Compression
When a summary is provided:
1. Original commands are preserved for execution
2. Detailed explanations are replaced by the summary
3. Token usage is significantly reduced
4. Essential context is maintained for future interactions

## Task Tracking System

The agent maintains state about ongoing tasks:

### Task Lifecycle
1. Task initiated by user prompt
2. Commands generated and executed
3. Results analyzed
4. Follow-up commands generated automatically
5. Process repeats until task completion

### Autonomous Operation
With the `:CONTINUE:` marker, the agent can execute multi-step sequences:
1. Execute initial commands
2. Analyze results
3. Generate follow-up commands
4. Continue until task is completed with `:TASK_COMPLETE:`

### Error Handling
- Command execution errors are captured
- Errors are fed back to the LLM for analysis
- Recovery strategies can be generated

## Logging System


Comprehensive logging captures all aspects of agent operation:

- Command execution and responses (flipper_agent_with_rag.py lines 179-203)
- LLM interactions (LLMAgent class)
- Context management decisions (estimate_tokens and _prune_history_by_tokens)
- Errors and exceptions (try/except blocks throughout)

Logs are timestamped and stored in the `logs` directory (configured in flipper_agent_with_rag.py lines 21-23) using rotating file handlers.

## Build & Execution

The system uses two key shell scripts:

### setup.sh
- Creates Python virtual environment
- Installs dependencies from requirements.txt
- Initializes PyFlipper submodule
- Sets executable permissions

### run.sh
- Handles command-line arguments
- Manages documentation vector store
- Configures LLM parameters
- Launches main agent with proper environment

## Dependencies

- PyFlipper (submodule for hardware communication)
- LangChain (RAG pipeline)
- FAISS (vector store)
- HuggingFace Embeddings (text embeddings)
- LiteLLM (LLM API abstraction)