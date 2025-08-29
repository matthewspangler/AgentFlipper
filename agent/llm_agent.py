import re
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union

# Get the AI logger configured in main.py
ai_logger = logging.getLogger("AgentFlipperAI")

from agent.agent_state import AgentState
from agent.llm_response_parser import LLMResponseParser
# Import litellm for LLM API calls
from litellm import completion, acompletion
from agent import prompts

class UnifiedLLMAgent:
    def __init__(self, config: Dict[str, Any], agent_state: AgentState):
        self.config = config
        self.agent_state = agent_state # Instance of AgentState
        self.provider = config.get("llm", {}).get("provider", "ollama")
        self.model = config.get("llm", {}).get("model", "default-model")
        # System prompt will likely come from prompts.py
        self.system_prompt = self._load_system_prompt()
        
        # LLM interaction details will be needed here (e.g., api_base)
        self.api_base = config.get("llm", {}).get("api_base", None)
        
        # Initialize the response parser
        self.response_parser = LLMResponseParser()

    def _load_system_prompt(self) -> str:
        """Load the appropriate system prompt for the agent."""
        return prompts.BASE_PROMPT
        
    async def create_initial_plan(self, task_description: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate the initial plan (sequence of actions) for a user task."""
        ai_logger.info(f"Creating initial plan for task: {task_description}")
        # Get conversation history from state for context
        history = self.agent_state.get_full_context_for_llm()

        # Generate prompt using a helper from prompts.py
        prompt = prompts.planning_prompt(task_description, history)
        ai_logger.debug(f"Prompt for initial plan:\n{prompt}")

        # Call the LLM
        llm_response = await self._call_llm(prompt, history)
        ai_logger.debug(f"Raw LLM response for initial plan:\n{llm_response}")

        # Parse the LLM's response into a structured plan
        plan = self.response_parser.parse_plan_response(llm_response)
        ai_logger.info(f"Parsed plan from LLM:\n{json.dumps(plan, indent=2)}")
        
        # The parser already handles the awaiting_human_input case,
        # so we can just return the plan directly
        return plan

    async def reflect_and_plan_next(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on a task result and determine next steps.
        Returns a dictionary indicating the next step:
        {"type": "add_tasks", "tasks": [...]}, {"type": "task_complete"}, or {"type": "awaiting_human_input", "question": "..."}, {"type": "error", "message": "..."}, {"type": "info", "information": "..."}
        """
        ai_logger.info(f"Reflecting on task {task.get('id')} result.")
        ai_logger.debug(f"Task:\n{json.dumps(task, indent=2)}")
        ai_logger.debug(f"Result:\n{json.dumps(result, indent=2)}")
        # Get conversation history and current state from agent_state
        history = self.agent_state.get_full_context_for_llm()

        # Generate prompt for reflection
        prompt = prompts.reflection_prompt(task, result, history)
        ai_logger.debug(f"Prompt for reflection:\n{prompt}")

        # Call the LLM
        llm_response = await self._call_llm(prompt, history)
        ai_logger.debug(f"Raw LLM response for reflection:\n{llm_response}")

        # Parse the LLM's response into a structured action/decision
        action = self.response_parser.parse_reflection_response(llm_response)
        ai_logger.info(f"Parsed reflection action from LLM:\n{json.dumps(action, indent=2)}")
        
        return action # Should be Dict with "type" key

    async def process_human_input_with_llm(self, human_input_result: Dict[str, Any]) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Process human input with the LLM to generate subsequent actions."""
        # Get relevant context from state
        history = self.agent_state.get_full_context_for_llm()
        
        # Generate prompt using human input and context
        task_context = human_input_result.get("request", {})
        human_input = human_input_result.get("response", "")
        prompt = prompts.human_input_processing_prompt(task_context, human_input, history)
        ai_logger.debug(f"Prompt for reflection:\n{prompt}")

        # Call the LLM
        llm_response = await self._call_llm(prompt, history)
        ai_logger.debug(f"Raw LLM response for reflection:\n{llm_response}")

        # Parse the LLM's response into actions or a completion signal
        action = self.response_parser.parse_reflection_response(llm_response) # Can reuse reflection parser if formats are similar
        
        return action # Should be List[Dict] or Dict with "type" key

    async def _call_llm(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> Any:
        """
        Make an API call to the LLM using LiteLLM.
        
        Args:
            prompt: The prompt text to send to the LLM
            history: Optional conversation history to provide context
            
        Returns:
            Response text from LLM (typically JSON string)
        """
        ai_logger.debug(f"Preparing LLM call with {self.provider}/{self.model}")
        
        # Format messages for the LLM
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history if provided
        if history:
            messages.extend(history)
        
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Log the prompt for debugging (but truncate very long prompts)
        truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
        ai_logger.debug(f"LLM Prompt: {truncated_prompt}")
        
        # Define the tool schema for function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "pyflipper",
                    "description": "Execute commands on the Flipper Zero device using the pyFlipper library",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commands": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of commands to execute on the Flipper Zero"
                            }
                        },
                        "required": ["commands"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_information",
                    "description": "Provide information to the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "information": {
                                "type": "string",
                                "description": "Information to display to the user"
                            }
                        },
                        "required": ["information"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_human",
                    "description": "Ask the human user a question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask the human"
                            }
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_complete",
                    "description": "Mark the current task as complete",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
        
        # Set up API parameters from config
        llm_config = self.config.get("llm", {})
        api_params = {
            "api_base": self.api_base or llm_config.get("api_base"),
            "temperature": llm_config.get("temperature", 0.3),
            "max_tokens": llm_config.get("max_tokens", 1000),
            "timeout": llm_config.get("timeout", 300)
        }
        
        # Clean up None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        # Format the full model name
        if self.provider == "ollama":
            full_model = f"ollama/{self.model}"
        else:
            full_model = f"{self.provider}/{self.model}"
            
        start_time = time.time()
        try:
            # Make the actual API call with error handling
            ai_logger.info(f"Making LLM call to {full_model}")
            
            # Debug logging for API parameters
            ai_logger.debug(f"API Parameters: {json.dumps(api_params, default=str)}")
            
            response = await acompletion( # Changed to acompletion for asynchronous call
                model=full_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **api_params
            )
            
            # Calculate and log latency
            latency = time.time() - start_time
            ai_logger.info(f"LLM call completed in {latency:.2f}s")
            
            # Log the raw response structure (abbreviated) for debugging
            ai_logger.debug(f"Raw LLM response type: {type(response)}")
            
            # Process response based on what the LLM returned
            content = ""
            tool_calls = []
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                
                # Handle tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls_output = []
                    for call in message.tool_calls:
                        try:
                            args = json.loads(call.function.arguments)
                            tool_calls_output.append({
                                "type": call.function.name,
                                "parameters": args
                            })
                        except json.JSONDecodeError as e:
                            ai_logger.error(f"Failed to parse tool arguments: {e}", exc_info=True)
                    
                    # Return a structured JSON response
                    content = json.dumps(tool_calls_output)
                    ai_logger.info(f"Extracted {len(tool_calls_output)} tool calls from LLM response")
                    
                # Handle standard text completion if no tool calls
                elif hasattr(message, 'content') and message.content:
                    content = message.content
                    ai_logger.info(f"Extracted text content from LLM response (length: {len(content)})")
                
                # Fall back to string representation if needed
                else:
                    content = str(message)
                    ai_logger.warning(f"Using string representation of message: {content[:100]}...")
            else:
                # Handle unexpected response format
                content = str(response)
                ai_logger.warning(f"Unexpected response format. Using string representation: {content[:100]}...")
            
            return content
            
        except Exception as e:
            # Log detailed error information
            error_duration = time.time() - start_time
            ai_logger.error(f"Error calling LLM after {error_duration:.2f}s: {str(e)}", exc_info=True)
            
            ai_logger.warning("LLM call failed! Using generic fallback response")
            return '[{"type": "provide_information", "parameters": {"information": "I encountered an error connecting to the LLM. Please try again or check your connection."}}]'


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in the given text.
    This is a rough approximation - about 4 characters per token.
    """
    return len(text) // 4
