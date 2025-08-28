import json
from typing import Dict, Any, Optional, List

# Shared base prompt with just identity and context (no tools list)
BASE_PROMPT = """You are an AI assistant that helps control a Flipper Zero device via its command-line interface.
Interpret the user's requests into specific Flipper Zero CLI commands.
You operate in a Plan, Act, and Reflect cycle."""

def planning_prompt(task_description: str, history: List[Dict[str, str]]) -> str:
    """Generate prompt for the planning phase."""
    # history is a list of messages {"role": "...", "content": "..."}
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-5:]]) # Example: last 5 messages
    
    return f"""{BASE_PROMPT}

Available Tools:
- pyflipper: Sends commands to the Flipper Zero device using the pyFlipper library. Parameters: {{"commands": ["list", "of", "commands"]}}
- provide_information: Displays information to the user. Parameters: {{"information": "string"}}
- ask_question: Requests input or decision from the human user. Parameters: {{"question": "string"}}

When asked to plan, provide a JSON array of tool calls.
When providing a plan, respond with a JSON array like:
[
    {{
        "type": "pyflipper",
        "parameters": {{
                "commands": [
                    "led bl 0",
                    "led bl 255"
                    ]
            }}
    }},
    {{
        "type": "provide_information",
        "parameters": {{
            "information": "Turned on backlight."
            }}
    }}
]

Previous Conversation History:
{history_text}

User Request: "{task_description}"

Based on the user's request and the conversation history, create a detailed plan (a JSON array of type objects) to accomplish this task using the available tools.
Focus on the next logical steps.
If you need more information from the user before creating a plan,
respond ONLY with the JSON for an 'awaiting_human_input' response: {{"type": "awaiting_human_input", "question": "Your specific question here"}}
"""
    
def reflection_prompt(task: Dict[str, Any], result: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    """Generate prompt for the reflection phase."""
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-5:]]) # Example: last 5 messages

    return f"""{BASE_PROMPT}

Available Tools:
- pyflipper: Sends commands to the Flipper Zero device using the pyFlipper library. Parameters: {{"commands": ["list", "of", "commands"]}}
- provide_information: Displays information to the user. Parameters: {{"information": "string"}}
- ask_question: Requests input or decision from the human user. Parameters: {{"question": "string"}}

When asked to reflect on a result, evaluate the outcome and decide the next step: provide new actions, signal task complete, or indicate that human input is required.

IMPORTANT: Do NOT use tools for state transitions like completing tasks. Instead, use reflection responses below.

When reflecting, if the task is complete, respond with:
{{"type": "task_complete"}}

When reflecting, if human input is needed, respond with:
{{"type": "awaiting_human_input", "question": "What color LED should I turn on?"}}

When reflecting, if more actions are needed, respond with:
{{"type": "add_tasks", "tasks": [...]}} # JSON array of type objects

When reflecting, if you just have information to provide:
{{"type": "info", "information": "The battery level is 85%."}}

Previous Conversation History:
{history_text}

You just executed the following task and received this result:
Task: {json.dumps(task, indent=2)}
Result: {json.dumps(result, indent=2)}

Based on this result, evaluate the outcome and decide the very next step towards completing the overall task goal (mentioned in the conversation history).
If there are remaining tasks in the queue (check context), continue executing them unless the current result indicates a final failure or completion. Only signal task_complete if the queue is empty AND the goal is met.
Respond with a JSON object following one of these types:
- {{"type": "task_complete"}} (If the overall task is finished)
- {{"type": "awaiting_human_input", "question": "Your question or request to the human"}} (If human input is needed to proceed)
- {{"type": "add_tasks", "tasks": [...]}} (If more actions are needed - provide a JSON array of type objects)
- {{"type": "info", "information": "Information to display to the user"}} (If the result provides information but no action is needed immediately)

Choose only ONE of these response types. Do not include any other text before or after the JSON.
"""
    
def human_input_processing_prompt(task_context: Dict[str, Any], human_input: str, history: List[Dict[str, str]]) -> str:
    """Generate prompt for processing human input."""
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-5:]]) # Example: last 5 messages

    return f"""{BASE_PROMPT}

Available Tools:
- pyflipper: Sends commands to the Flipper Zero device using the pyFlipper library. Parameters: {{"commands": ["list", "of", "commands"]}}
- provide_information: Displays information to the user. Parameters: {{"information": "string"}}
- ask_question: Requests input or decision from the human user. Parameters: {{"question": "string"}}

When processing human input, determine the appropriate next step based on the input received.
You can either:
1. Provide a new plan (JSON array of tool calls)
2. Signal that the task is complete
3. Ask for more clarification

Previous Conversation History:
{history_text}

The human user just provided the following input:
Human Input: "{human_input}"
Original Request Context: {json.dumps(task_context.get("human_input_request", {}), indent=2)}

Based on this human input and the overall task goal (in history), provide the next steps as a JSON array of type objects.
If the human input indicates the task is complete, respond with {{"type": "task_complete"}}.
If the human input requires further clarification, respond with {{"type": "awaiting_human_input", "question": "Your question here"}}.
"""