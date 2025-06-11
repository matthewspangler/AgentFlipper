import re
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union

# Get the AI logger configured in main.py
ai_logger = logging.getLogger("AgentFlipperAI")

# Assuming AgentState is available
# from ..agent_loop.agent_state import AgentState

# Import litellm for LLM API calls
from litellm import completion, acompletion # Import acompletion

class UnifiedLLMAgent:
    def __init__(self, config: Dict[str, Any], agent_state: Any): # Use Any for AgentState for now
        self.config = config
        self.agent_state = agent_state # Instance of AgentState
        self.provider = config.get("llm", {}).get("provider", "ollama")
        self.model = config.get("llm", {}).get("model", "default-model")
        # System prompt will likely come from prompts.py
        self.system_prompt = self._load_system_prompt() 
        
        # LLM interaction details will be needed here (e.g., api_base)
        self.api_base = config.get("llm", {}).get("api_base", None)

    def _load_system_prompt(self) -> str:
        """Load the appropriate system prompt for the agent."""
        # In a real implementation, load this from prompts.py or config
        # For now, hardcode or load a basic one
        return """You are an AI assistant that helps control a Flipper Zero device via its command-line interface.
        Your primary function is to interpret user requests and translate them into precise Flipper Zero CLI commands that are then executed via a serial connection using the pyFlipper library.

        **CRITICAL INSTRUCTIONS:**

        1.  **Prioritize Provided Documentation (RAG Context):** Always refer to and prioritize the provided Flipper Zero CLI documentation when determining the correct command and syntax for a user's request. The commands in this documentation are the ONLY valid commands you can execute.
        2.  **Use `pyflipper` Tool:** To send commands to the Flipper Zero, you MUST use the `pyflipper` tool. Provide the exact, correct CLI command (e.g., `led bl 255`, not `power backlight on`) as found in the documentation.
        3.  **Analyze Command Results (Reflection):** After a command is executed, you will receive the device's response in the reflection step. You MUST analyze this response.
            *   If the response indicates success, you may proceed to the next step or mark the task complete if the overall goal is achieved.
            *   If the response indicates an error (e.g., "Usage:", "ERROR:", "illegal option"), the command failed. You MUST NOT mark the task complete. Instead, you should:
                *   Re-evaluate the command based on the documentation and the error message.
                *   If possible, generate a corrected command using `pyflipper`.
                *   If unsure how to correct the command or the error is severe, use the `ask_human` tool to request assistance.
        4.  **Plan, Act, Reflect Cycle:** You operate in a Plan, Act, and Reflect cycle.
            *   **Plan:** Generate a JSON array of tool calls to achieve the user's request.
            *   **Act:** The system executes your tool calls.
            *   **Reflect:** You analyze the results and determine the next step (add more tasks, ask human, mark complete).
        5.  **Use Available Tools:** You have the following tools:
            *   `pyflipper`: For sending CLI commands to the Flipper Zero serial.
            *   `provide_information`: For displaying helpful text to the user.
            *   `ask_human`: For asking the user questions or requesting help.
            *   `mark_task_complete`: To signal that the *overall* user request is fully satisfied.

        **Remember:** Your goal is to successfully execute the user's intent on the Flipper Zero by generating correct commands based on the documentation and reacting appropriately to command results. Do not invent commands or guess syntax.
        """
        
    async def create_initial_plan(self, task_description: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate the initial plan (sequence of actions) for a user task."""
        ai_logger.info(f"Creating initial plan for task: {task_description}")
        # Get conversation history from state for context
        # history = self.agent_state.get_full_context_for_llm() # Need this method
        history = self.agent_state.conversation_history # Placeholder

        # Generate prompt using a helper from prompts.py
        # prompt = prompts.planning_prompt(task_description, history) # Need prompts module

        # Basic planning prompt for now
        prompt = f"""
        User Request: "{task_description}"

        Based on this request and previous conversation history, create a plan to accomplish this task using the available tools.
        Respond with a JSON array of action objects.
        If you need more information from the user before creating a plan,
        respond with a single action: {{"action": "ask_human", "parameters": {{"question": "Your question here"}}}}
        """
        ai_logger.debug(f"Prompt for initial plan:\n{prompt}")

        # Call the LLM
        # llm_response = await self._call_llm(prompt, history) # Need history parameter in _call_llm
        llm_response = await self._call_llm(prompt) # Placeholder call
        ai_logger.debug(f"Raw LLM response for initial plan:\n{llm_response}")

        # Parse the LLM's response into a structured plan
        plan = self._parse_plan_from_response(llm_response)
        ai_logger.info(f"Parsed plan from LLM:\n{json.dumps(plan, indent=2)}")
        
        # Check if the parsed plan contains a request for human input
        if self._plan_needs_human_input(plan):
            question = self._extract_question_from_plan(plan)
            # LLM wants human input to *plan*. Signal this back to the loop.
            return {"type": "awaiting_human_input", "question": question} # Return a specific structure to AgentLoop

        # Assume the parsed response is the plan (list of actions)
        return plan # Should be List[Dict]

    async def reflect_and_plan_next(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on a task result and determine next steps.
        Returns a dictionary indicating the next action:
        {"type": "add_tasks", "tasks": [...]}, {"type": "task_complete"}, or {"type": "awaiting_human_input", "question": "..."}, {"type": "error", "message": "..."}, {"type": "info", "information": "..."}
        """
        ai_logger.info(f"Reflecting on task {task.get('id')} result.")
        ai_logger.debug(f"Task:\n{json.dumps(task, indent=2)}")
        ai_logger.debug(f"Result:\n{json.dumps(result, indent=2)}")
        # Get conversation history and current state from agent_state
        # history = self.agent_state.get_full_context_for_llm() # Need this method
        history = self.agent_state.conversation_history # Placeholder

        # Generate prompt for reflection
        # prompt = prompts.reflection_prompt(task, result, history) # Need prompts module

        # Basic reflection prompt for now
        prompt = f"""
        You previously executed a task and got the following result:
        Task: {json.dumps(task, indent=2)}
        Result: {json.dumps(result, indent=2)}

        Based on this result and the overall task goal (in history), decide the next step:
        1. If the overall task is now complete, respond with a single action: {{"type": "task_complete"}}
        2. If further actions are needed to achieve the goal, provide a JSON array of those action objects.
        3. If the result is ambiguous, indicates an error requiring human help, or you need more information to proceed, respond with: {{"type": "awaiting_human_input", "question": "Your question/request to the human"}}
        4. If the result provides information the user should see but no further action or human input is needed immediately, respond with: {{"type": "info", "information": "Information to display"}}
        """
        ai_logger.debug(f"Prompt for reflection:\n{prompt}")

        # Call the LLM
        # llm_response = await self._call_llm(prompt, history) # Need history parameter
        llm_response = await self._call_llm(prompt) # Placeholder call
        ai_logger.debug(f"Raw LLM response for reflection:\n{llm_response}")

        # Parse the LLM's response into a structured action/decision
        action = self._parse_reflection_response(llm_response)
        ai_logger.info(f"Parsed reflection action from LLM:\n{json.dumps(action, indent=2)}")
        
        return action # Should be Dict with "type" key

    # NOTE: The implementation plan also included process_human_input here.
    # This could be kept here or moved to AgentLoop or HumanInteractionHandler
    # depending on where the logic for *processing* human input with the LLM fits best.
    # Keeping it in LLM Agent makes sense if LLM needs to process the input to make a new plan.
    async def process_human_input_with_llm(self, human_input_result: Dict[str, Any]) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Process human input with the LLM to generate subsequent actions."""
        # Get relevant context from state
        # history = self.agent_state.get_full_context_for_llm() # Need this method
        # task_context = human_input_result.get("request", {}).get("context") # Could store context with the request
        
        # Generate prompt using human input and context
        # prompt = prompts.human_input_processing_prompt(task_context, human_input_result["response"]) # Need prompts module
        
        # Basic prompt for processing human input
        prompt = f"""
        The human provided the following input in response to a request:
        Human Input: {human_input_result.get("response")}
        Original Request: {json.dumps(human_input_result.get("request", {}), indent=2)}

        Based on this input and the conversation history, provide the next steps as a JSON array of action objects.
        If the human input indicates the task is complete, respond with {{"type": "task_complete"}}.
        If the human input requires further clarification, respond with {{"type": "awaiting_human_input", "question": "..."}}.
        """
        ai_logger.debug(f"Prompt for reflection:\n{prompt}")

        # Call the LLM
        # llm_response = await self._call_llm(prompt, history) # Need history parameter
        llm_response = await self._call_llm(prompt) # Placeholder call
        ai_logger.debug(f"Raw LLM response for reflection:\n{llm_response}")

        # Parse the LLM's response into actions or a completion signal
        action = self._parse_reflection_response(llm_response) # Can reuse reflection parser if formats are similar
        
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
            "timeout": llm_config.get("timeout", 60)
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
                                "action": call.function.name,
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
            return '[{"action": "provide_information", "parameters": {"information": "I encountered an error connecting to the LLM. Please try again or check your connection."}}]'
            # The rest of the code in this method remains the same

    def _parse_plan_from_response(self, response_text: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract structured plan (list of actions) from LLM text response."""
        ai_logger.debug(f"Attempting to parse LLM plan response:\n{response_text}")

        try:
            parsed_response = json.loads(response_text)
            
            if isinstance(parsed_response, list):
                ai_logger.info(f"Successfully parsed plan as JSON array with {len(parsed_response)} actions.")
                
                for item in parsed_response:
                    if "action" not in item and "name" in item:
                        item["action"] = item.pop("name")
                        ai_logger.debug(f"Renamed 'name' key to 'action' for consistent format")
                        
                    if item.get("action") == "ask_human":
                        ai_logger.info("Converting ask_human action to awaiting_human_input format")
                        return {
                            "type": "awaiting_human_input",
                            "question": item.get("parameters", {}).get("question", "Need more information")
                        }
                    
                    if item.get("action") == "execute_commands":
                        item["action"] = "pyflipper"
                        ai_logger.debug("Converted 'execute_commands' action to 'pyflipper'")
                
                return parsed_response

            elif isinstance(parsed_response, dict) and "type" in parsed_response:
                ai_logger.info(f"Parsed response as structured dict with type: {parsed_response['type']}")
                return parsed_response

            else:
                ai_logger.warning(f"Parsed JSON is not a recognized format: {parsed_response}")
                return {"type": "invalid_plan", "message": "LLM response was not a JSON array."}

        except json.JSONDecodeError:
            ai_logger.debug("Direct JSON parsing failed, attempting regex extraction")
            
            try:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_string = json_match.group(0)
                    ai_logger.debug(f"Extracted JSON string using regex:\n{json_string}")
                    
                    plan = json.loads(json_string)
                    if isinstance(plan, list):
                        ai_logger.info(f"Successfully parsed plan as JSON array using regex.")
                        
                        for item in plan:
                            if "action" not in item and "name" in item:
                                item["action"] = item.pop("name")
                                ai_logger.debug(f"Renamed 'name' key to 'action' for consistent format")
                            
                            if item.get("action") == "ask_human":
                                ai_logger.info("Converting ask_human action to awaiting_human_input format")
                                return {
                                    "type": "awaiting_human_input",
                                    "question": item.get("parameters", {}).get("question", "Need more information")
                                }
                            
                            if item.get("action") == "execute_commands":
                                item["action"] = "pyflipper"
                                ai_logger.debug("Converted 'execute_commands' action to 'pyflipper'")
                        
                        return plan

                    else:
                        ai_logger.warning(f"Parsed JSON is not a list: {plan}")
                        return {"type": "invalid_plan", "message": "LLM response was not a JSON array."}

                dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if dict_match:
                    json_string = dict_match.group(0)
                    ai_logger.debug(f"Extracted JSON dict using regex:\n{json_string}")
                    
                    parsed_dict = json.loads(json_string)
                    if isinstance(parsed_dict, dict):
                        if parsed_dict.get("action") == "ask_human":
                            return {
                                "type": "awaiting_human_input",
                                "question": parsed_dict.get("parameters", {}).get("question", "Need more information")
                            }
                        
                        if "type" in parsed_dict:
                            return parsed_dict

                ai_logger.error(f"Could not parse plan or structured response from LLM response:\n{response_text}")
                return {"type": "parsing_failed", "message": "LLM response did not contain a valid plan or structured action."}

            except json.JSONDecodeError as e:
                ai_logger.error(f"JSON Decode Error parsing LLM plan response: {e}", exc_info=True)
                return {"type": "parsing_error", "message": f"Failed to parse JSON: {e}", "raw_response": response_text}
            except Exception as e:
                ai_logger.error(f"Unexpected error during plan parsing: {e}", exc_info=True)
                return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}"}
        except Exception as e:
            ai_logger.error(f"Unexpected error during plan parsing: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}"}

    def _parse_reflection_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the reflection response from the LLM into a structured action.
        Expected LLM response types: {"type": "task_complete"}, {"type": "awaiting_human_input", "question": "..."},
        {"type": "add_tasks", "tasks": [...]}, {"type": "info", "information": "..."}, etc.
        """
        ai_logger.debug(f"Attempting to parse LLM reflection response:\n{response_text}")

        # First try direct JSON parsing for properly formatted responses
        try:
            # This should work with _call_llm direct output
            parsed_response = json.loads(response_text)

            # Handle case where the response is a list of action objects
            # (possible from tool calls in the new implementation)
            if isinstance(parsed_response, list):
                ai_logger.debug("Parsed response is a list - looking for specific actions to convert")

                # Check for special cases that should convert to specific reflection types
                for item in parsed_response:
                    # Normalize action name if needed
                    if "action" not in item and "name" in item:
                        item["action"] = item.pop("name")
                        ai_logger.debug(f"Renamed 'name' key to 'action' for consistent format")

                    # Convert mark_task_complete to task_complete response
                    if item.get("action") == "mark_task_complete":
                        ai_logger.info("Converting mark_task_complete action to task_complete reflection")
                        return {"type": "task_complete"}

                    # Convert ask_human to awaiting_human_input
                    if item.get("action") == "ask_human":
                        ai_logger.info("Converting ask_human action to awaiting_human_input reflection")
                        return {
                            "type": "awaiting_human_input",
                            "question": item.get("parameters", {}).get("question", "Need more information")
                        }
                    # Check if action is the old execute_commands name and convert
                    if item.get("action") == "execute_commands":
                        item["action"] = "pyflipper"
                        ai_logger.debug("Converted 'execute_commands' action to 'pyflipper'")


                # If it's a list but not one of the special cases,
                # return it as an add_tasks reflection
                ai_logger.info("Converting list of actions to add_tasks reflection")
                return {"type": "add_tasks", "tasks": parsed_response}

            # Handle case where parsed response is already a reflection dict with "type" key
            if isinstance(parsed_response, dict) and "type" in parsed_response:
                # Validate known types
                valid_types = ["task_complete", "awaiting_human_input", "add_tasks", "info"]
                if parsed_response["type"] in valid_types:
                    # Validate required parameters for specific types
                    if parsed_response["type"] == "awaiting_human_input" and "question" not in parsed_response:
                        ai_logger.warning("'awaiting_human_input' missing 'question' field")
                        parsed_response["question"] = "Need more information to proceed. Can you provide details?"

                    if parsed_response["type"] == "add_tasks" and ("tasks" not in parsed_response or not isinstance(parsed_response["tasks"], list)):
                        ai_logger.warning("'add_tasks' missing or invalid 'tasks' list")
                        return {"type": "unhandled_reflection", "message": "'add_tasks' missing or invalid tasks list"}

                    if parsed_response["type"] == "info" and "information" not in parsed_response:
                        ai_logger.warning("'info' missing 'information' field")
                        return {"type": "unhandled_reflection", "message": "'info' missing information field"}

                    ai_logger.info(f"Successfully parsed reflection with type: {parsed_response['type']}")
                    return parsed_response
                else:
                    ai_logger.warning(f"Unknown reflection type: {parsed_response['type']}")
                    return {"type": "unhandled_reflection", "message": f"Unknown type: {parsed_response['type']}"}
        except json.JSONDecodeError:
            ai_logger.debug("Direct JSON parsing failed, attempting regex extraction")

        # Fall back to regex extraction if direct parsing fails
        try:
            # First try to find a JSON dictionary that might contain a reflection
            dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if dict_match:
                json_string = dict_match.group(0)
                ai_logger.debug(f"Extracted JSON dict using regex:\n{json_string}")

                action = json.loads(json_string)
                if isinstance(action, dict):
                    # If it has a "type" key, treat it as a reflection
                    if "type" in action:
                        valid_types = ["task_complete", "awaiting_human_input", "add_tasks", "info"]
                        if action["type"] in valid_types:
                            ai_logger.info(f"Successfully parsed reflection action from regex: {action['type']}")
                            return action

                    # If it has an "action" key, it might be a tool call to convert
                    if "action" in action:
                        # Check if action is the old execute_commands name and convert
                        if action.get("action") == "execute_commands":
                            action["action"] = "pyflipper"
                            ai_logger.debug("Converted 'execute_commands' action to 'pyflipper'")

                        if action["action"] == "mark_task_complete":
                            return {"type": "task_complete"}
                        elif action["action"] == "ask_human" or action["action"] == "ask_question":
                            return {
                                "type": "awaiting_human_input",
                                "question": action.get("parameters", {}).get("question", "Need more information")
                            }

            # If no reflection dict found, check for a JSON array that might be a plan
            array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if array_match:
                json_string = array_match.group(0)
                ai_logger.debug(f"Extracted JSON array using regex:\n{json_string}")

                array_data = json.loads(json_string)
                if isinstance(array_data, list) and len(array_data) > 0:
                    ai_logger.info(f"Found a list of {len(array_data)} actions - converting to add_tasks")
                    return {"type": "add_tasks", "tasks": array_data}

            # If we couldn't find valid JSON, try to extract a task completion signal
            if "task complete" in response_text.lower() or "task is complete" in response_text.lower():
                ai_logger.info("Found task completion phrase in text")
                return {"type": "task_complete"}

            ai_logger.warning(f"Could not extract structured reflection from: {response_text}")
            return {"type": "unhandled_reflection", "message": "Could not parse reflection", "raw_response": response_text}

        except json.JSONDecodeError as e:
            ai_logger.error(f"JSON Decode Error parsing LLM reflection response: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Failed to parse JSON: {e}", "raw_response": response_text}
        except Exception as e:
            ai_logger.error(f"Unexpected error during reflection parsing: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}", "raw_response": response_text}


    def _plan_needs_human_input(self, plan: Union[List[Dict[str, Any]], Dict[str, Any]]) -> bool:
        """Determine if the parsed plan or response indicates a request for human input."""
        # Check if the plan is the specific {"type": "awaiting_human_input"} structure
        return isinstance(plan, dict) and plan.get("type") == "awaiting_human_input"

    def _extract_question_from_plan(self, plan: Dict[str, Any]) -> str:
        """Extract the question to ask the human from the specific structure."""
        return plan.get("question", "Need more information to proceed. Can you provide details?")

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in the given text.
    This is a rough approximation - about 4 characters per token.
    """
    return len(text) // 4


class LLMAgent:
    """Class for the LLM Agent."""

    def __init__(self, config: Dict[str, Any], rag_retriever: RAGRetriever = None):
        """Initialize the LLM agent with the given configuration"""
        self.config = config # Store the raw config dictionary

        # Safely get llm config, defaulting to an empty dict if not present
        llm_config = config.get("llm", {})

        self.provider = llm_config.get("provider", "ollama")
        self.model_name = llm_config.get("model", "qwen2.5-coder:14b")
        self.api_base = llm_config.get("api_base", "http://localhost:11434")
        self.max_history_tokens = llm_config.get("max_history_tokens", 2000)

        # Fixed system prompt with explicit completion requirements
        self.system_prompt = """You are an assistant controlling a Flipper Zero device through its CLI.
You need to intelligently determine when to provide commands versus information.

Use the following tools to structure your responses:

1. execute_commands: For sending commands to the Flipper Zero device
2. provide_information: For displaying information to the user
3. ask_question: For asking the user for clarification
4. mark_task_complete: To indicate the task is finished

You MUST use these tools to structure your responses. Do not use any special markers.

The system will automatically process tool calls in the order they are provided.

CRITICAL: After completing the user's request, you MUST call mark_task_complete to return control to the user.
Without this, the system will loop indefinitely.

IMPORTANT: For every command execution, you MUST include mark_task_complete in the same response.
The only exceptions are when you need to ask a question or provide additional information.

RULES:
1. When you have COMPLETELY satisfied the user's request, ALWAYS include mark_task_complete
2. If you need to execute multiple commands to satisfy the request, do so before marking complete
3. Only ask questions when you need clarification to complete the task
4. Provide information when it helps the user understand what was done

Examples:

For command execution with completion:
[
  {
    "name": "execute_commands",
    "arguments": {
      "commands": ["led bl 0"]
    }
  },
  {
    "name": "mark_task_complete",
    "arguments": {}
  }
]

For information display with completion:
[
  {
    "name": "provide_information",
    "arguments": {
      "information": "Backlight turned off"
    }
  },
  {
    "name": "mark_task_complete",
    "arguments": {}
  }
]

Available commands include: info, gpio, ibutton, irda, lfrfid, nfc, subghz, usb,
vibro, update, bt, storage, bad_usb, backlight, led, and others.

IMPORTANT: The backlight command is 'led bl <value>' where value is 0-255."""

        # User configurable prompt that adds to the system prompt
        self.user_prompt = llm_config.get("user_prompt", "")
        self.rag_retriever = rag_retriever

        # Initialize conversation history
        self.conversation_history = []
        # Get max_recursion_depth from llm_config
        self.max_recursion_depth = llm_config.get("max_recursion_depth", 10)
        self.task_in_progress = False
        self.task_description = ""

        # Format the model name for LiteLLM
        if self.provider == "ollama":
            self.full_model = f"ollama/{self.model_name}"
        else:
            # Handle other providers if needed
            self.full_model = f"{self.provider}/{self.model_name}"

    async def get_commands(self, user_input: str, previous_results: Optional[List[Tuple[str, str]]] = None) -> List[dict]:
        """
        Get Flipper Zero commands from the LLM based on user input and previous results.

        Args:
            user_input: The original user query or task description
            previous_results: Optional list of previous command results as (command, response) tuples

        Returns:
            List of tool call objects (each is a dict with 'name' and 'arguments')
        """
        try:
            # Prepare API parameters
            # Get api_base from the config dict
            llm_config = self.config.get("llm", {})
            api_params = {"api_base": llm_config.get("api_base", "http://localhost:11434")}

            # Retrieve relevant documentation if RAG is enabled
            context = ""
            if self.rag_retriever and self.rag_retriever.initialized:
                # Combine the user_input with any previous error context for better RAG
                search_query = user_input
                if previous_results:
                    # Add info about errors to the query to help find relevant docs
                    for cmd, resp in previous_results:
                        if "illegal option" in resp or "error" in resp.lower() or "usage:" in resp:
                            search_query += f" {cmd} error {resp}"

                docs = self.rag_retriever.retrieve(search_query) # Assuming retrieve is synchronous
                if docs:
                    context = "Here is relevant information about Flipper Zero CLI commands:\n\n"
                    context += "\n\n".join(docs)
                    logger.info(f"Retrieved {len(docs)} relevant documents")

            # Build the enhanced system prompt with user prompt and retrieved context
            enhanced_prompt = self.system_prompt

            # Add user prompt if specified
            if self.user_prompt:
                enhanced_prompt += f"\n\n{self.user_prompt}"

            # Add context if available
            if context:
                enhanced_prompt += "\n\n" + context

            # If this is a new task, store it
            if not self.task_in_progress and not previous_results:
                self.task_in_progress = True
                self.task_description = user_input
                logger.info(f"Starting new task: {user_input}")
            else:
                logger.info(f"Continuing task: {self.task_description}")

            # Prepare messages with conversation history
            messages = [{"role": "system", "content": enhanced_prompt}]

            # Add conversation history
            messages.extend(self.conversation_history)

            # Add previous command results if any
            if previous_results:
                result_content = "Here are the results of the previous commands:\n\n"
                for cmd, resp in previous_results:
                    result_content += f"Command: {cmd}\nResponse: {resp}\n\n"

                # Add context for next action
                if any("error" in resp.lower() for cmd, resp in previous_results):
                    result_content += "Based on these results, provide the next tool calls to continue the task."
                else:
                    result_content += "The command executed successfully. You MUST include mark_task_complete in your response to prevent an infinite loop."

                messages.append({"role": "user", "content": result_content})
                logger.info("Added previous command results to prompt")

            # Add the current user input if not just continuing from previous commands
            if not previous_results:
                messages.append({"role": "user", "content": user_input})

            logger.info(f"Sending request to {self.full_model}")
            print(f"{Colors.PURPLE}Thinking...{Colors.ENDC}")  # Keep yellow

            try:
                # Use function calling with tool definitions
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_commands",
                            "description": "Execute commands on the Flipper Zero device",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "commands": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of commands to execute"
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
                            "name": "ask_question",
                            "description": "Ask the user a question",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "Question to ask the user"
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
                            "description": "Mark the task as complete",
                            "parameters": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    }
                ]

                response = completion( # Await removed - check for blocking behavior
                    model=self.full_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=500,
                    **api_params
                )

                # Extract tool calls from response
                tool_calls = []
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    message = response.choices[0].message
                    if hasattr(message, 'tool_calls'):
                        for call in message.tool_calls:
                            try:
                                # Parse arguments as JSON
                                arguments = json.loads(call.function.arguments)
                                tool_calls.append({
                                    "name": call.function.name,
                                    "arguments": arguments
                                })
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")
                                ai_logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")

                logger.info(f"Generated {len(tool_calls)} tool calls")

                # Log AI response for debugging
                ai_logger.info("===== AI RESPONSE =====")
                ai_logger.info(f"Model: {self.full_model}")
                ai_logger.info(f"Input Messages: {json.dumps(messages, indent=2)}")
                ai_logger.info(f"Raw Response: {response}")
                ai_logger.info(f"Tool Calls: {json.dumps(tool_calls, indent=2)}")
                ai_logger.info("======================")

                # Validate tool calls format
                if tool_calls:
                    # Check if mark_task_complete is missing after execute_commands
                    has_execute = any(call['name'] == 'execute_commands' for call in tool_calls)
                    has_mark = any(call['name'] == 'mark_task_complete' for call in tool_calls)

                    if has_execute and not has_mark:
                        # Add mark_task_complete if missing
                        tool_calls.append({
                            "name": "mark_task_complete",
                            "arguments": {}
                        })
                        logger.warning("Added missing mark_task_complete tool call")

                return tool_calls

            except Exception as e:
                error_msg = f"An error occurred during LLM processing."
                # Log the full traceback for debugging
                logger.error("LLM processing error:", exc_info=True)
                ai_logger.error("LLM processing error:", exc_info=True)
                # Return a generic error message to the user
                return [{"name": "provide_information", "arguments": {"information": error_msg}}]

        except Exception as e:
            error_msg = f"Error generating tool calls: {str(e)}"
            logger.error(error_msg, exc_info=True)
            ai_logger.error(error_msg, exc_info=True)
            return [{"name": "provide_information", "arguments": {"information": error_msg}}]

    def _prune_history_by_tokens(self):
        """Prune conversation history to stay under the token limit"""
        if not self.conversation_history:
            return

        # Calculate current token count
        total_tokens = 0
        for message in self.conversation_history:
            total_tokens += estimate_tokens(message["content"])

        # If under limit, no action needed
        if total_tokens <= self.max_history_tokens:
            return

        # Log before pruning
        logger.info(f"History token count ({total_tokens}) exceeds limit ({self.max_history_tokens}), pruning...")

        # Remove oldest messages until under token limit
        while total_tokens > self.max_history_tokens and len(self.conversation_history) > 2:
            # Remove oldest exchange (user + assistant message)
            removed_user = self.conversation_history.pop(0)
            removed_assistant = self.conversation_history.pop(0)

            # Recalculate token count
            total_tokens -= estimate_tokens(removed_user["content"])
            total_tokens -= estimate_tokens(removed_assistant["content"])

        logger.info(f"After pruning: {len(self.conversation_history)} messages, ~{total_tokens} tokens")

    async def generate_summary(self) -> str:
        """
        Generate a summary of the conversation so far.
        This is called by the Python script rather than relying on the LLM to provide summaries.
        Called automatically on TASK_COMPLETE or after N iterations.

        Returns:
            Summary string
        """
        # Check if there's meaningful content to summarize
        if not self.conversation_history:
            return "Task just started, no progress to summarize yet."

        # Get the last few exchanges for context
        recent_history = self.conversation_history[-8:] if len(self.conversation_history) >= 8 else self.conversation_history

        # Format the conversation for summarization
        summary_prompt = "Please provide a brief summary of what has been accomplished so far:\n\n"

        # Add the recent history
        for message in recent_history:
            role = message["role"]
            content = message["content"]
            summary_prompt += f"{role.upper()}: {content}\n\n"

        # Add the request for summary
        summary_prompt += "Summary of progress so far:"

        try:
            # Use the same LLM to generate a summary
            response = await completion( # Await the LiteLLM completion call
                model=self.full_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the progress of the task so far in a concise paragraph."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=150,
                # Get api_base from the config dict
                # Move this logic outside the completion call
                api_base=self.config.get("llm", {}).get("api_base", "http://localhost:11434")
            )

            # Extract the response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                summary = response.choices[0].message.content
            else:
                summary = str(response)

            logger.info(f"Generated summary: {summary}")
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Unable to generate summary: {str(e)}"

    def add_execution_results_to_history(self, results: List[Tuple[str, str]]):
        """
        Add command execution results to the conversation history.
        This ensures the summary generation has access to what commands were run
        and what their results were.

        Args:
            results: List of (command, response) tuples from command execution
        """
        if not results:
            return

        # Format the results into a readable message
        result_content = "Command execution results:\n\n"
        for cmd, resp in results:
            result_content += f"Command: {cmd}\nResponse: {resp}\n\n"

        # Add as a system message to conversation history
        self.conversation_history.append({"role": "system", "content": result_content})
        logger.info("Added execution results to conversation history for summarization")