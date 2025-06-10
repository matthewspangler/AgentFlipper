import re
import asyncio
import json
import logging # Import logging
from typing import Dict, Any, Optional, List, Union

# Get the AI logger configured in main.py
ai_logger = logging.getLogger("AgentFlipperAI")

# Assuming AgentState is available
# from ..agent_loop.agent_state import AgentState

# Assuming litellm is installed
# from litellm import completion

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
        Interpret the user's requests into specific Flipper Zero CLI commands.
        You operate in a Plan, Act, and Reflect cycle.
        When asked to plan, provide a JSON array of actions (tool calls).
        When asked to reflect on a result, evaluate the outcome and decide the next step: provide new actions, signal task complete, or indicate that human input is required.
        Use the available tools as instructed.
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

        # Call the LLM
        # llm_response = await self._call_llm(prompt, history) # Need history parameter
        llm_response = await self._call_llm(prompt) # Placeholder call

        # Parse the LLM's response into actions or a completion signal
        action = self._parse_reflection_response(llm_response) # Can reuse reflection parser if formats are similar
        
        return action # Should be List[Dict] or Dict with "type" key

    async def _call_llm(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> Any:
        """Make the actual LLM API call (placeholder)."""
        # This is where litellm.completion would be called
        # Need to format prompt and history into messages list
        # Need to handle API key, base URL, model name from config
        # Need to handle potential errors from the API call
        print(f"--- Calling LLM ---")
        print(f"Prompt:\n{prompt}")
        # print(f"History:\n{json.dumps(history, indent=2)}") # If history is used
        print(f"-------------------")

        # Placeholder: In a real implementation, await litellm.completion(...)
        # For now, return dummy responses for testing the loop structure
        await asyncio.sleep(1) # Simulate LLM latency

        # Enhanced dummy responses for development:
       # Enhanced dummy responses for development based on prompt content:
        # Enhanced dummy responses for development based on prompt content:
        if "Reflect on a task result" in prompt.lower():
             # Dummy reflection: return a dictionary indicating next action
            ai_logger.debug("Dummy LLM: Responding to reflection prompt with 'task_complete'")
            return '{"type": "task_complete"}' # Return a JSON string representing a dictionary
        elif "Based on this human input" in prompt.lower():
             # Dummy response to human input: signal complete
            ai_logger.debug("Dummy LLM: Responding to human input prompt with 'task_complete'")
            return '{"type": "task_complete"}' # Return a JSON string representing a dictionary
        else:
            # Assume it's a planning request (initial plan or adding tasks after reflection)
            ai_logger.debug("Dummy LLM: Assuming planning prompt, returning a plan.")
            if "backlight" in prompt.lower():
                return '[{"action": "execute_commands", "parameters": {"commands": ["led bl 255"]}}, {"action": "provide_information", "parameters": {"message": "Backlight turned on successfully."}}]' # JSON string representing a list
            else:
                # Default planning response - always return a valid plan format
                return '[{"action": "execute_commands", "parameters": {"commands": ["info"]}}, {"action": "provide_information", "parameters": {"message": "Command executed."}}]' # JSON string representing a list


    def _parse_plan_from_response(self, response_text: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract structured plan (list of actions) from LLM text response."""
        ai_logger.debug(f"Attempting to parse LLM plan response:\n{response_text}")
        # This is a critical parsing step. Need to handle LLM responses
        # that might include conversational text before or after the JSON.
        # Use regex or careful string parsing to extract the JSON array.
        # Use regex to find the JSON array within the response text.
        # LLMs can sometimes include conversational text or formatting around the JSON output,
        # so direct json.loads() may fail. This regex attempts to extract the JSON array string.


        try:
            # Attempt to find and parse JSON array from the response text
            # Attempt to find and parse JSON array from the response text
            # Using a greedy match (`.*`) to capture the whole array assuming no nested top-level arrays.
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                ai_logger.debug(f"Extracted JSON string using regex:\n{json_string}")
                # Need to be careful with escaped characters or malformed JSON from LLM
                plan = json.loads(json_string)
                if isinstance(plan, list):
                    ai_logger.info(f"Successfully parsed plan as JSON array.")
                    return plan
                else:
                    # print(f"Parsed JSON is not a list: {plan}") # Replace with proper logging
                    ai_logger.warning(f"Parsed JSON is not a list: {plan}")
                    return {"type": "invalid_plan", "message": "LLM response was not a JSON array."}

            # If no JSON array found, check for other structured responses like ask_human
            try:
                 json_dict = json.loads(response_text)
                 if isinstance(json_dict, dict) and json_dict.get("type") == "awaiting_human_input":
                     ai_logger.info(f"Parsed response as 'awaiting_human_input'.")
                     return json_dict # LLM returned structured human input request instead of plan
            except json.JSONDecodeError:
                 pass # Not a structured dict

            # print(f"Could not parse plan or structured response from LLM response: {response_text}") # Log parsing failure
            ai_logger.error(f"Could not parse plan or structured response from LLM response:\n{response_text}")
            return {"type": "parsing_failed", "message": "LLM response did not contain a valid plan or structured action."}

        except json.JSONDecodeError as e:
            # print(f"JSON Decode Error parsing LLM plan response: {e}") # Log parsing error
            ai_logger.error(f"JSON Decode Error parsing LLM plan response: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Failed to parse LLM plan response: {e}"}
        except Exception as e:
            # print(f"Unexpected error during plan parsing: {e}") # Log other errors
            ai_logger.error(f"Unexpected error during plan parsing: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}"}


    def _parse_reflection_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the reflection response from the LLM into a structured action.
        Expected LLM response types: {"type": "task_complete"}, {"type": "awaiting_human_input", "question": "..."}, {"type": "add_tasks", "tasks": ...]}} etc.
        """
        ai_logger.debug(f"Attempting to parse LLM reflection response:\n{response_text}")
        try:
            # Attempt to parse the response as JSON
            action = json.loads(response_text)

            # Validate the structure
            if not isinstance(action, dict) or "type" not in action:
                 ai_logger.warning(f"Invalid reflection format from LLM: {response_text}")
                 return {"type": "unhandled_reflection", "message": "Invalid format", "raw_response": response_text} # Include raw response

            # Validate known types
            valid_types = ["task_complete", "awaiting_human_input", "add_tasks", "info"]
            if action["type"] not in valid_types:
                 ai_logger.warning(f"Unknown reflection type from LLM: {action['type']}")
                 return {"type": "unhandled_reflection", "message": f"Unknown type: {action['type']}", "raw_response": response_text} # Include raw response

            # Basic validation for types with parameters
            if action["type"] == "awaiting_human_input" and "question" not in action:
                 ai_logger.warning(f"'awaiting_human_input' missing 'question': {response_text}")
                 return {"type": "unhandled_reflection", "message": "'awaiting_human_input' missing question", "raw_response": response_text} # Include raw response
            if action["type"] == "add_tasks":
                 if "tasks" not in action or not isinstance(action["tasks"], list):
                     ai_logger.warning(f"'add_tasks' missing or invalid 'tasks' list: {response_text}")
                     return {"type": "unhandled_reflection", "message": "'add_tasks' missing or invalid tasks list", "raw_response": response_text} # Include raw response
                 # Optionally validate tasks structure within the list

            ai_logger.info(f"Successfully parsed reflection action: {action.get('type')}")
            return action

        except json.JSONDecodeError as e:
            ai_logger.error(f"JSON Decode Error parsing LLM reflection response: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Failed to parse JSON: {e}", "raw_response": response_text} # Include raw response
        except Exception as e:
            ai_logger.error(f"Unexpected error during reflection parsing: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}", "raw_response": response_text} # Include raw response


    def _plan_needs_human_input(self, plan: Union[List[Dict[str, Any]], Dict[str, Any]]) -> bool:
        """Determine if the parsed plan or response indicates a request for human input."""
        # Check if the plan is the specific {"type": "awaiting_human_input"} structure
        return isinstance(plan, dict) and plan.get("type") == "awaiting_human_input"

    def _extract_question_from_plan(self, plan: Dict[str, Any]) -> str:
        """Extract the question to ask the human from the specific structure."""
        return plan.get("question", "Need more information to proceed. Can you provide details?")