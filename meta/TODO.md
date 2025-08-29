# AgentFlipper Improvement Roadmap

This document outlines the planned improvements for the AgentFlipper project, prioritized by their impact on stability, maintainability, and extensibility.

## 1. Core Architectural Refinements

**Reasoning**: These are the highest priority because they address foundational structural issues that affect the entire codebase. Completing these first will provide a more stable and maintainable foundation for future development.

- [x] **Consolidate the Agent Loop**: Refactor `agent_loop.py` and `agent_loop_processor.py` into a single, definitive `AgentLoop` class to eliminate redundancy.
- [ ] **Refactor the LLM Agent**: Extract the complex response parsing logic from `llm_agent.py` into a separate `LLMResponseParser` class to improve modularity.
- [ ] **Refine Component Initialization**: Refactor the initialization of UI-dependent components in `main.py` to make dependencies more explicit and avoid reliance on a mutable `app_instance` variable.
- [ ] **Standardize Tool Naming**: Standardize on a single tool name convention (e.g., "ask_human") across the codebase to improve consistency and reduce errors.

## 2. Add Testing Infrastructure

**Reasoning**: A robust testing suite is essential for long-term project health. It enables you to make changes with confidence and catch regressions early. This should be a high priority after the initial architectural cleanup.

- [ ] Set up pytest framework with a clear directory structure.
- [ ] Create unit tests for core components (`AgentState`, `TaskManager`, etc.).
- [ ] Add integration tests for LLM response parsing.
- [ ] Implement mock classes for LLM and hardware interactions to isolate tests.
- [ ] Create test fixtures for common test scenarios.
- [ ] Add a CI/CD pipeline (e.g., GitHub Actions) for automated testing.

## 3. Dynamic Tool Registration & Extensibility

**Reasoning**: Making the tool system extensible is a high-impact feature that directly aligns with the goals of creating a flexible AI agent. This is the next logical step after stabilizing the core architecture.

- [ ] **Enhance Tool Extensibility**: Implement a dynamic tool registration system (e.g., a `tools` directory with a registry) to replace the hardcoded tool list in `llm_agent.py`.
- [ ] Design a plugin architecture for tools.
- [ ] Implement a tool registry with runtime registration capability.
- [ ] Create a base `Tool` class with a standard interface.
- [ ] Update `ToolExecutor` to use the registry.
- [ ] Add the capability to enable/disable tools dynamically.
- [ ] Document the plugin system for future tool developers.

## 4. Advanced State Management Refactoring

**Reasoning**: While the current `AgentState` is functional, refactoring it will improve scalability and modularity as the agent's capabilities grow. This is a good medium-priority task.

- [ ] Split `AgentState` into more focused components:
  - [ ] `ConversationHistory` class for managing dialog.
  - [ ] `TaskState` class for managing the current execution state.
  - [ ] `ContextBuffer` class for managing working memory.
- [ ] Update all components that interact with `AgentState`.
- [ ] Ensure backward compatibility during the transition.
- [ ] Update documentation to reflect the new state management architecture.

## 5. Agent Persistence

**Reasoning**: This is a valuable feature for user experience, allowing users to resume their work. It's best to implement this after the core architecture and state management are more mature.

- [ ] Design a serialization format for the agent's state.
- [ ] Implement save/load functionality for conversation history.
- [ ] Add persistence for the task queue's contents.
- [ ] Create mechanisms to restore the execution context.
- [ ] Add CLI options for loading previous sessions.
- [ ] Document the persistence capabilities and limitations.

## 6. Special Markers System

**Reasoning**: This is a more specialized feature that can be implemented once the core functionality is robust and well-tested.

- [ ] Define the purpose and design of the Special Markers System.
- [ ] Document the system in the `ARCHITECTURE.md` file.
- [ ] Implement marker detection in relevant components (likely `agent_loop.py`).
- [ ] Add marker generation to LLM prompts where appropriate.
- [ ] Create tests to verify that marker processing works correctly.

## 7. Allow executing natural english scripts

## 8. Come up with end to end testing

## 9. Bad vibe code that hardcodes responses to LLM output

For example, from llm_response_parser.py:
```
# Look for phrases indicating the main goal was achieved
completion_indicators = [
    "primary goal was accomplished",
    "main objective was achieved",
    "core task was completed",
    "successfully turned on and off",
    "overall task was completed"
]
```
This doesn't need to be hardcoded because what the LLM spits out can be highly variable. Figuring out natural language shouldn't be hardcoded, LLM's should be used to make these sorts of decisions.

I found little mistakes like this buried a lot throughout the project. Need to instruct the LLM to not do this, and fix them.