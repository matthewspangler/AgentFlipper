import json
from typing import Dict, Any, Optional, List

# Define the main system prompt
SYSTEM_PROMPT = """You are an AI assistant that helps control a Flipper Zero device via its command-line interface.
Interpret the user's requests into specific Flipper Zero CLI commands.
You operate in a Plan, Act, and Reflect cycle.
When asked to plan, provide a JSON array of actions (tool calls).
When asked to reflect on a result, evaluate the outcome and decide the next step: provide new actions, signal task complete, or indicate that human input is required.
Use the available tools as instructed.

Available Tools:
- pyflipper: Sends commands to the Flipper Zero device using the pyFlipper library. Parameters: {{"commands": ["list", "of", "strings"]}}
- provide_information: Displays information to the user. Parameters: {{"information": "string"}}
- ask_human: Requests input or decision from the human user. Parameters: {{"question": "string"}}
- mark_task_complete: Signals that the overall task is finished. Parameters: {{}}

When providing a plan or new actions, respond with a JSON array like:
[
    {{"action": "pyflipper", "parameters": {{"commands": ["info", "led bl 255"]}}}},
    {{"action": "provide_information", "parameters": {{"information": "Turned on backlight."}}}},
    {{"action": "mark_task_complete", "parameters": {{}}}}
]

When reflecting, if the task is complete, respond with:
{{"type": "task_complete"}}

When reflecting, if human input is needed, respond with:
{{"type": "awaiting_human_input", "question": "What color LED should I turn on?"}}

When reflecting, if more actions are needed, respond with:
{{"type": "add_tasks", "tasks": [...]}} # JSON array of action objects

When reflecting, if you just have information to provide:
{{"type": "info", "information": "The battery level is 85%."}}
"""

def planning_prompt(task_description: str, history: List[Dict[str, str]]) -> str:
    """Generate prompt for the planning phase."""
    # history is a list of messages {"role": "...", "content": "..."}
    # You might want to format history appropriately for the prompt
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-5:]]) # Example: last 5 messages
    
    return f"""
{SYSTEM_PROMPT}

Previous Conversation History:
{history_text}

User Request: "{task_description}"

Based on the user's request and the conversation history, create a detailed plan (a JSON array of action objects) to accomplish this task using the available tools.
Focus on the next logical steps.
If you need more information from the user before creating a plan,
respond ONLY with the JSON for an 'awaiting_human_input' action: {{"type": "awaiting_human_input", "question": "Your specific question here"}}
"""
    
def reflection_prompt(task: Dict[str, Any], result: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    """Generate prompt for the reflection phase."""
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-5:]]) # Example: last 5 messages

    return f"""
{SYSTEM_PROMPT}

Previous Conversation History:
{history_text}

You just executed the following task and received this result:
Task: {json.dumps(task, indent=2)}
Result: {json.dumps(result, indent=2)}

Based on this result, evaluate the outcome and decide the very next step towards completing the overall task goal (mentioned in the conversation history).
Respond with a JSON object following one of these types:
- {{"type": "task_complete"}} (If the overall task is finished)
- {{"type": "awaiting_human_input", "question": "Your question or request to the human"}} (If human input is needed to proceed)
- {{"type": "add_tasks", "tasks": [...]}} (If more actions are needed - provide a JSON array of action objects)
- {{"type": "info", "information": "Information to display to the user"}} (If the result provides information but no action is needed immediately)

Choose only ONE of these response types. Do not include any other text before or after the JSON.
"""
    
def human_input_processing_prompt(task_context: Dict[str, Any], human_input: str, history: List[Dict[str, str]]) -> str:
    """Generate prompt for processing human input."""
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-5:]]) # Example: last 5 messages

    return f"""
{SYSTEM_PROMPT}

Previous Conversation History:
{history_text}

The human user just provided the following input:
Human Input: "{human_input}"
Original Request Context: {json.dumps(task_context.get("human_input_request", {}), indent=2)}

Based on this human input and the overall task goal (in history), provide the next steps as a JSON array of action objects.
If the human input indicates the task is complete, respond with {{"type": "task_complete"}}.
If the human input requires further clarification, respond with {{"type": "awaiting_human_input", "question": "Your question here"}}.
"""