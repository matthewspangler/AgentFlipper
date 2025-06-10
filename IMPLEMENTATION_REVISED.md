# AgentFlipper Implementation Plan (Revised)

This document outlines the implementation approach for the Plan and Act pattern with integrated human-in-the-loop capabilities.

## Simplified Code Organization

```
flipper-agent/
  ├── agent_loop/            # Core agent loop components
  │   ├── __init__.py
  │   ├── agent_loop.py      # Main loop implementation
  │   ├── agent_state.py     # Centralized state management
  │   ├── task_manager.py    # Task queue handling
  │   └── tool_executor.py   # Tool execution logic
  ├── llm/                   # LLM interaction components
  │   ├── __init__.py
  │   ├── llm_agent.py       # Unified agent for planning and reflection
  │   └── prompts.py         # Centralized prompt templates
  ├── ui/                    # UI components
  │   ├── __init__.py  
  │   ├── textual_app.py     # Main Textual UI
  │   └── human_interaction.py # Human-in-the-loop UI components
  └── main.py                # Application entry point
```

## Key Classes and Their Responsibilities

### 1. `AgentState`

Central repository for all state data, making debugging easier by providing a single source of truth.

```python
class AgentState:
    def __init__(self, config=None):
        self.conversation_history = []
        self.context_buffer = {}
        self.task_queue = []
        self.current_task = None
        self.task_results = {}
        self.config = config or {}
        self.flipper_agent = None
        self.llm_agent = None
        self.awaiting_human_input = False
        self.human_input_request = None
        
    def add_to_history(self, entry):
        """Add entry to conversation history with optional pruning."""
        pass
        
    def update_context(self, key, value):
        """Update a value in the context buffer."""
        pass
        
    def clear_for_new_session(self):
        """Reset state for a new user session."""
        pass
        
    def request_human_input(self, request_type, prompt, options=None):
        """Set state to await human input with given prompt."""
        self.awaiting_human_input = True
        self.human_input_request = {
            "type": request_type,  # "question", "approval", "selection", etc.
            "prompt": prompt,
            "options": options,    # Optional list of choices
            "timestamp": time.time()
        }
        
    def process_human_input(self, input_value):
        """Process received human input and update state."""
        self.awaiting_human_input = False
        # Record the interaction in history
        self.add_to_history({
            "role": "human",
            "content": f"Response to {self.human_input_request['type']}: {input_value}"
        })
        result = {"request": self.human_input_request, "response": input_value}
        self.human_input_request = None
        return result
```

### 2. `TaskManager`

Handles all task queue operations with clear semantics for queue operations.

```python
class TaskManager:
    def __init__(self, agent_state):
        self.agent_state = agent_state
        
    def add_user_task(self, user_input):
        """Convert user input to a task and add to queue."""
        task = self._create_task_from_user_input(user_input)
        self.add_task(task)
        
    def add_plan_to_queue(self, plan):
        """Add a sequence of tasks from a plan to the queue."""
        for task_data in plan:
            task = self._create_task_from_plan_item(task_data)
            self.add_task(task)
            
    def add_task(self, task):
        """Add a single task to the queue."""
        self.agent_state.task_queue.append(task)
        
    def get_next_task(self):
        """Get and remove the next task from the queue."""
        if self.is_empty():
            return None
        self.agent_state.current_task = self.agent_state.task_queue.pop(0)
        return self.agent_state.current_task
        
    def is_empty(self):
        """Check if task queue is empty."""
        return len(self.agent_state.task_queue) == 0
```

### 3. `UnifiedLLMAgent`

A simplified approach where planning and reflection are combined in a single agent.

```python
class UnifiedLLMAgent:
    def __init__(self, config, agent_state):
        self.config = config
        self.agent_state = agent_state
        self.provider = config.get("llm", {}).get("provider", "ollama")
        self.model = config.get("llm", {}).get("model", "default-model")
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self):
        """Load the appropriate system prompt for the agent."""
        # Could load from a file or config
        return """You are an AI assistant that helps control a Flipper Zero device.
        You create plans to accomplish tasks and reflect on the results of actions.
        When appropriate, ask the human for clarification or approval."""
        
    async def create_initial_plan(self, task_description):
        """Generate the initial plan for a user task."""
        prompt = f"""
        Based on the user's request: "{task_description}", 
        create a detailed plan with specific steps to accomplish this task.
        Each step should be a valid tool call with appropriate parameters.
        Format your response as a JSON array of action objects.
        
        If you need more information from the user before creating a plan,
        respond with a "ask_human" action instead.
        """
        
        plan_response = await self._call_llm(prompt)
        plan = self._parse_plan_from_response(plan_response)
        
        # Check if the plan contains a request for human input
        if self._plan_needs_human_input(plan):
            question = self._extract_question_from_plan(plan)
            self.agent_state.request_human_input("question", question)
            return {"type": "awaiting_human_input"}
            
        return plan
        
    async def reflect_and_plan_next(self, task, result):
        """Reflect on a task result and determine next steps."""
        prompt = f"""
        You executed: {json.dumps(task)}
        Result: {json.dumps(result)}
        
        Based on this result:
        1. Does this require human attention or feedback? If so, specify what to ask.
        2. Should we add more tasks to the plan? If so, detail them.
        3. Is the overall task complete? If so, state "TASK_COMPLETE".
        
        Format your response appropriately based on your decision.
        """
        
        reflection = await self._call_llm(prompt)
        action = self._parse_reflection(reflection)
        
        # Handle different outcome types
        if action["type"] == "ask_human":
            self.agent_state.request_human_input("feedback", action["question"])
            return {"type": "awaiting_human_input"}
            
        return action
        
    async def process_human_input(self, task_context, human_input):
        """Process human input and determine next steps."""
        prompt = f"""
        Task context: {json.dumps(task_context)}
        Human input: {human_input}
        
        Based on this human input, what steps should be taken next?
        Format your response as a plan of action steps.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_plan_from_response(response)
        
    async def _call_llm(self, prompt, system_message=None):
        """Make the actual LLM API call."""
        # Implementation details for calling the LLM
        pass
        
    def _parse_plan_from_response(self, response):
        """Extract structured plan from LLM response."""
        # Implementation to parse response into structured format
        pass
        
    def _parse_reflection(self, response):
        """Parse the reflection response into an action."""
        # Implementation to determine the next action based on reflection
        pass
        
    def _plan_needs_human_input(self, plan):
        """Determine if the plan contains a request for human input."""
        if not plan or not isinstance(plan, list):
            return False
        return any(item.get("action") == "ask_human" for item in plan)
        
    def _extract_question_from_plan(self, plan):
        """Extract the question to ask the human from the plan."""
        for item in plan:
            if item.get("action") == "ask_human":
                return item.get("parameters", {}).get("question", "Need more information to proceed. Can you provide details?")
        return "Need more information to proceed. Can you provide details?"
```

### 4. Prompt Management (`llm/prompts.py`)

Centralize and manage all LLM prompts, potentially using templates for flexibility and maintainability.

```python
# Example structure in prompts.py

SYSTEM_PROMPT = """You are an AI assistant...""" # Define the main system prompt

def planning_prompt(task_description):
    """Generate prompt for the planning phase."""
    return f"""Based on the user's request: "{task_description}", ..."""
    
def reflection_prompt(task, result):
    """Generate prompt for the reflection phase."""
    return f"""You executed: {json.dumps(task)}\nResult: {json.dumps(result)}\n..."""
    
def human_input_processing_prompt(task_context, human_input):
    """Generate prompt for processing human input."""
    return f"""Task context: {json.dumps(task_context)}\nHuman input: {human_input}\n..."""
    
# Add other specific prompt functions as needed
```

### 5. `HumanInteractionHandler`

Dedicated component for managing human-in-the-loop interactions.

```python
class HumanInteractionHandler:
    def __init__(self, agent_state, app_instance):
        self.agent_state = agent_state
        self.app_instance = app_instance
        
    async def request_input(self, request_type, prompt, options=None):
        """Request input from the human and wait for response."""
        self.agent_state.request_human_input(request_type, prompt, options)
        
        # Display the request in the UI
        await self.app_instance.display_human_request(
            request_type, 
            prompt, 
            options
        )
        
        # The UI will handle updating the agent_state when input is received
        # and the main loop will continue processing
        
    async def handle_approval_request(self, plan):
        """Request approval for a plan before execution."""
        plan_display = self._format_plan_for_display(plan)
        prompt = f"Would you like to approve this plan?\n\n{plan_display}"
        
        return await self.request_input("approval", prompt, ["Approve", "Reject", "Modify"])
        
    async def handle_clarification(self, question):
        """Request clarification from human."""
        return await self.request_input("question", question)
        
    async def handle_error_resolution(self, error, task):
        """Request human help for error resolution."""
        prompt = f"An error occurred during task execution:\n{error}\n\nHow would you like to proceed?"
        options = ["Retry", "Skip", "Abort", "Provide solution"]
        
        return await self.request_input("error_resolution", prompt, options)
        
    def _format_plan_for_display(self, plan):
        """Format a plan nicely for human display."""
        # Implementation to format the plan steps clearly
        result = "Plan Steps:\n"
        for i, step in enumerate(plan, 1):
            result += f"{i}. {step.get('action', 'Unknown')}"
            if "parameters" in step:
                params = step["parameters"]
                result += f": {json.dumps(params, indent=2)}"
            result += "\n"
        return result
```

### 5. `AgentLoop`

The main control flow implementation with human-in-the-loop integration.

```python
class AgentLoop:
    def __init__(self, agent_state, task_manager, tool_executor, llm_agent, human_interaction, app_instance):
        self.agent_state = agent_state
        self.task_manager = task_manager
        self.tool_executor = tool_executor
        self.llm_agent = llm_agent
        self.human_interaction = human_interaction
        self.app_instance = app_instance
        
    async def process_user_request(self, user_input):
        """Entry point for processing a user request."""
        # Initialize a new session
        await self.app_instance.display_message(f"Processing: '{user_input}'")
        
        # Add user input to task queue
        self.task_manager.add_user_task(user_input)
        
        # Initialize/update context buffer
        self._initialize_context_buffer()
        
        # Start the main loop
        await self._run_main_loop()
        
    async def _run_main_loop(self):
        """Main agent loop with human-in-the-loop integration."""
        while True:
            # Check if we're waiting for human input
            if self.agent_state.awaiting_human_input:
                # This state will be updated by the UI when human input is received
                # The loop will continue on the next iteration
                await asyncio.sleep(0.1)  # Short sleep to avoid CPU spinning
                continue
                
            # Check if any tasks remain
            if self.task_manager.is_empty():
                # Generate initial plan from LLM if needed
                plan = await self.llm_agent.create_initial_plan(
                    self.agent_state.context_buffer.get("current_goal")
                )
                
                # Check if the plan requires human input
                if plan and plan.get("type") == "awaiting_human_input":
                    # The state is already set to await human input
                    # We'll process it on the next loop iteration
                    continue
                    
                # Request human approval for the plan if configured
                if self.agent_state.config.get("require_plan_approval", False):
                    await self.human_interaction.handle_approval_request(plan)
                    # This will set awaiting_human_input to True
                    # We'll process the approval on the next loop iteration
                    continue
                
                # Add approved or auto-approved plan to task queue
                self.task_manager.add_plan_to_queue(plan)
            
            # Check if task queue is now empty (could happen if planning failed)
            if self.task_manager.is_empty():
                # No more tasks, end the loop
                break
                
            # Get next task from queue
            task = self.task_manager.get_next_task()
            
            # Execute the task
            result = await self.tool_executor.execute_task(task)
            
            # Store result in state
            self.agent_state.task_results[task["id"]] = result
            
            # If task execution failed, consider asking for human help
            if not result.get("success", False):
                if self.agent_state.config.get("human_error_resolution", False):
                    await self.human_interaction.handle_error_resolution(
                        result.get("error", "Unknown error"),
                        task
                    )
                    continue
            
            # Give result to LLM for reflection and next steps
            reflection_action = await self.llm_agent.reflect_and_plan_next(task, result)
            
            # Check if reflection suggests asking the human
            if reflection_action.get("type") == "awaiting_human_input":
                # The state is already set to await human input
                # We'll process it on the next loop iteration
                continue
                
            # Handle reflection outcome
            if reflection_action.get("type") == "add_tasks":
                new_tasks = reflection_action.get("tasks", [])
                for task in new_tasks:
                    self.task_manager.add_task(task)
            elif reflection_action.get("type") == "task_complete":
                # Exit criteria met
                await self.app_instance.display_message("✅ Task complete")
                break
                
        # End of processing
        await self.app_instance.display_message("Processing complete")
        
    def _initialize_context_buffer(self):
        """Set up or update the context buffer."""
        # Implementation as before
        pass
```

## Human-in-the-Loop Integration Points

The revised design incorporates human involvement at several key points in the workflow:

1. **Initial Plan Creation**
   * When the LLM determines it needs more information to create a good plan
   * When the system is configured to require human approval for all plans

2. **Task Execution**
   * When a task execution fails and human help is enabled
   * When a specific tool is designed to require human input (e.g., `ask_question` tool)

3. **Reflection and Next Steps**
   * When the LLM's reflection determines human input would be valuable
   * When critical decisions need to be made about the next course of action

4. **Plan Approval**
   * Optionally requiring human approval before executing a plan
   * Allowing humans to review, modify or reject proposed plans

5. **Final Verification**
   * Optionally asking humans to verify that the task is complete
   * Requesting final confirmation before terminating the agent loop

## UI Integration for Human-in-the-Loop

The `AgentFlipper` Textual UI class will need additional components to handle human-in-the-loop interactions:

```python
class AgentFlipper(App):
    # ... existing UI code ...
    
    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = message.value.strip()
        self.input_widget.value = ""  # Clear the input field
        
        if not user_input:
            return
            
        # Check if we're awaiting human input in response to a question
        if self.agent_state.awaiting_human_input:
            # Process the human input through the state
            human_input_result = self.agent_state.process_human_input(user_input)
            request_type = human_input_result["request"]["type"]
            
            # Display acknowledgment of the input
            await self.display_message(f"[#2ed832]Received {request_type} response: {user_input}[/#2ed832]")
            
            # If we have a worker waiting for this input, no need to start a new one
            return
        
        # Regular user task input processing
        # ... existing code for handling new user requests ...

    ### UI-Agent Communication

The UI (`textual_app.py`) will communicate with the `AgentLoop` and `AgentState` to handle human input:

- **Requesting Input:** When the `AgentLoop` requires human input, it calls methods on the `HumanInteractionHandler`. The handler updates `AgentState` and signals the UI (e.g., by calling a method on `app_instance`).
- **Receiving Input:** The UI's input handling (`on_input_submitted`) checks if the agent is `awaiting_human_input` via `AgentState`. If so, it processes the input using `agent_state.process_human_input()` and potentially signals the `AgentLoop` to resume (e.g., by clearing the flag and using an `asyncio.Event` or similar mechanism to wake the loop if it was explicitly waiting).

    async def display_human_request(self, request_type, prompt, options=None):
        """Display a request for human input."""
        # Format the prompt
        formatted_prompt = f"[bold #ff9722]👤 {request_type.capitalize()} Required:[/]\n{prompt}\n"
        
        # Format options if provided
        if options:
            formatted_prompt += "\nOptions:\n"
            for i, option in enumerate(options, 1):
                formatted_prompt += f"  {i}. {option}\n"
                
        await self.display_message(formatted_prompt)
        
        # For approval requests, add extra formatting
        if request_type == "approval":
            await self.display_message("[#2ed832]Please type 'approve', 'reject', or 'modify'[/#2ed832]")
        elif options:
            await self.display_message("[#2ed832]Please type your choice (number or text)[/#2ed832]")
```

## Implementation Roadmap

1. **Phase 1**: Create core classes (AgentState, TaskManager)
2. **Phase 2**: Implement UnifiedLLMAgent for planning and reflection
3. **Phase 3**: Develop the HumanInteractionHandler and UI components
4. **Phase 4**: Implement the main AgentLoop with human-in-the-loop integration
5. **Phase 5**: Add the task visualization UI component

This architecture provides a more streamlined approach where planning and reflection are unified, with clear integration points for human-in-the-loop interactions throughout the agent's workflow.