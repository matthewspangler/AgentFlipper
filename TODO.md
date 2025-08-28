# AgentFlipper Improvement Roadmap

This document outlines the planned improvements for the AgentFlipper project based on architecture analysis and best practices from established AI agents.

## 1. Standardize Tool Naming

- [ ] Standardize on either "ask_human" or "ask_question" across the codebase
- [ ] Update tool name in `tool_executor.py` registration
- [ ] Update corresponding references in `llm_agent.py`
- [ ] Update tool names in prompt templates in `prompts.py`
- [ ] Ensure tool names in LLM parsing logic match the standardized names

## 2. Implement/Document Special Markers System

- [ ] Define the purpose and design of the Special Markers System
- [ ] Document the system in the ARCHITECTURE.md file
- [ ] Implement marker detection in relevant components (likely agent_loop.py)
- [ ] Add marker generation to LLM prompts where appropriate
- [ ] Create tests to verify marker processing works correctly

## 3. Add Testing Infrastructure

- [ ] Set up pytest framework with directory structure
- [ ] Create unit tests for core components (AgentState, TaskManager, etc.)
- [ ] Add integration tests for LLM response parsing
- [ ] Implement mock classes for LLM and hardware interactions
- [ ] Create test fixtures for common test scenarios
- [ ] Add Github Actions or similar CI/CD pipeline for automated testing

## 4. Further Refactor State Management

- [ ] Split AgentState into more focused components:
  - [ ] ConversationHistory class for managing dialog
  - [ ] TaskState class for managing current execution state
  - [ ] ContextBuffer class for managing working memory
- [ ] Update all components that interact with AgentState
- [ ] Ensure backward compatibility during transition
- [ ] Update documentation to reflect new state management

## 5. Dynamic Tool Registration

- [ ] Design a plugin architecture for tools
- [ ] Implement a tool registry with runtime registration capability
- [ ] Create a base Tool class with standard interface
- [ ] Update ToolExecutor to use the registry
- [ ] Add capability to enable/disable tools dynamically
- [ ] Document the plugin system for future tool developers

## 6. Agent Persistence

- [ ] Design serialization format for agent state
- [ ] Implement save/load functionality for conversation history
- [ ] Add persistence for task queue contents
- [ ] Create mechanisms to restore execution context
- [ ] Add CLI options for loading previous sessions
- [ ] Document persistence capabilities and limitations