import re
import json
import logging
from typing import Dict, Any, Union, List

# Get the AI logger configured in main.py
ai_logger = logging.getLogger("AgentFlipperAI")


class LLMResponseParser:
    """
    Handles parsing of LLM responses for both planning and reflection phases.
    Extracts structured data from various response formats.
    """
    
    def __init__(self):
        # Define special response types that don't get wrapped in lists
        self.special_response_types = [
            "awaiting_human_input", 
            "task_complete", 
            "error", 
            "info", 
            "invalid_plan", 
            "parsing_error", 
            "parsing_failed",
            "unhandled_reflection"
        ]
        
        # Valid reflection types for validation
        self.valid_reflection_types = [
            "task_complete", 
            "awaiting_human_input", 
            "add_tasks", 
            "info"
        ]
        
        # Task completion indicators for special case handling
        self.completion_indicators = [
            "primary goal was accomplished",
            "main objective was achieved",
            "core task was completed",
            "successfully turned on and off",
            "overall task was completed"
        ]
    
    def parse_plan_response(self, response_text: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract structured plan (list of actions) from LLM text response.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            Either a list of tool calls or a special response dict
        """
        ai_logger.debug(f"Attempting to parse LLM plan response:\n{response_text}")

        try:
            parsed_response = json.loads(response_text)
            return self._process_parsed_plan(parsed_response)
            
        except json.JSONDecodeError:
            ai_logger.debug("Direct JSON parsing failed, attempting regex extraction")
            return self._parse_plan_with_regex(response_text)
        except Exception as e:
            ai_logger.error(f"Unexpected error during plan parsing: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}"}

    def parse_reflection_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the reflection response from the LLM into a structured action.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            A dictionary with type and associated data
        """
        ai_logger.debug(f"Attempting to parse LLM reflection response:\n{response_text}")

        # First try direct JSON parsing for properly formatted responses
        try:
            parsed_response = json.loads(response_text)
            return self._process_parsed_reflection(parsed_response)
            
        except json.JSONDecodeError:
            ai_logger.debug("Direct JSON parsing failed, attempting regex extraction")
            return self._parse_reflection_with_regex(response_text)

    def _process_parsed_plan(self, parsed_response: Any) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Process successfully parsed JSON for plan responses."""
        if isinstance(parsed_response, list):
            ai_logger.info(f"Successfully parsed plan as JSON array with {len(parsed_response)} tool calls.")
            
            for item in parsed_response:
                self._normalize_tool_call(item)
                
                # Check for special ask_human type
                if item.get("type") == "ask_human":
                    ai_logger.info("Converting ask_human type to awaiting_human_input format")
                    return {
                        "type": "awaiting_human_input",
                        "question": item.get("parameters", {}).get("question", "Need more information")
                    }
            
            return parsed_response

        elif isinstance(parsed_response, dict) and "type" in parsed_response:
            ai_logger.info(f"Parsed response as structured dict with type: {parsed_response['type']}")
            
            # Handle add_tasks structure
            if parsed_response.get("type") == "add_tasks" and "tasks" in parsed_response:
                ai_logger.info("Extracting tasks from 'add_tasks' structure for initial plan.")
                return parsed_response["tasks"]
            
            # Handle special response types
            if parsed_response.get("type") in self.special_response_types:
                return parsed_response
            
            # Regular tool calls get wrapped in a list
            ai_logger.info(f"Wrapping single tool call with type '{parsed_response.get('type')}' in a list")
            return [parsed_response]
        
        elif isinstance(parsed_response, dict) and "action" in parsed_response:
            ai_logger.info("Parsed a single action object, converting to type and wrapping in a list.")
            parsed_response["type"] = parsed_response.pop("action")
            return [parsed_response]

        else:
            ai_logger.warning(f"Parsed JSON is not a recognized format: {parsed_response}")
            return {"type": "invalid_plan", "message": "LLM response was not a JSON array."}

    def _process_parsed_reflection(self, parsed_response: Any) -> Dict[str, Any]:
        """Process successfully parsed JSON for reflection responses."""
        # Handle list of tool calls
        if isinstance(parsed_response, list):
            ai_logger.debug("Parsed response is a list - looking for specific tool calls to convert")

            for item in parsed_response:
                self._normalize_tool_call(item)

                # Convert mark_task_complete to task_complete response
                if item.get("type") == "mark_task_complete":
                    ai_logger.info("Converting mark_task_complete type to task_complete reflection")
                    return {"type": "task_complete"}

                # Convert ask_human to awaiting_human_input
                if item.get("type") == "ask_human":
                    ai_logger.info("Converting ask_human type to awaiting_human_input reflection")
                    return {
                        "type": "awaiting_human_input",
                        "question": item.get("parameters", {}).get("question", "Need more information")
                    }

            # If it's a list but not one of the special cases, return as add_tasks
            ai_logger.info("Converting list of tool calls to add_tasks reflection")
            return {"type": "add_tasks", "tasks": parsed_response}

        # Handle structured reflection dict
        if isinstance(parsed_response, dict) and "type" in parsed_response:
            return self._validate_reflection_dict(parsed_response, parsed_response.get("type"))
        
        # Default case
        return {"type": "unhandled_reflection", "message": "Unexpected response format"}

    def _validate_reflection_dict(self, parsed_response: Dict[str, Any], response_type: str) -> Dict[str, Any]:
        """Validate and process a reflection dictionary."""
        if response_type in self.valid_reflection_types:
            # Validate required parameters
            if response_type == "awaiting_human_input" and "question" not in parsed_response:
                ai_logger.warning("'awaiting_human_input' missing 'question' field")
                parsed_response["question"] = "Need more information to proceed. Can you provide details?"

            if response_type == "add_tasks" and ("tasks" not in parsed_response or not isinstance(parsed_response["tasks"], list)):
                ai_logger.warning("'add_tasks' missing or invalid 'tasks' list")
                return {"type": "unhandled_reflection", "message": "'add_tasks' missing or invalid tasks list"}

            if response_type == "info" and "information" not in parsed_response:
                ai_logger.warning("'info' missing 'information' field")
                return {"type": "unhandled_reflection", "message": "'info' missing information field"}
            
            # Special case: Check if task is complete despite command failures
            if response_type == "awaiting_human_input":
                completion_check = self._check_task_completion_indicators(parsed_response, json.dumps(parsed_response))
                if completion_check:
                    return completion_check
            
            ai_logger.info(f"Successfully parsed reflection with type: {response_type}")
            return parsed_response
        else:
            ai_logger.warning(f"Unknown reflection type: {response_type}")
            return {"type": "unhandled_reflection", "message": f"Unknown type: {response_type}"}

    def _check_task_completion_indicators(self, parsed_response: Dict[str, Any], response_text: str) -> Union[Dict[str, Any], None]:
        """
        Check if the response indicates task completion despite command failures.
        Returns task_complete dict if indicators found, None otherwise.
        """
        if "question" in parsed_response:
            question = parsed_response["question"].lower()
            
            # Only intercept when the question is about a command failure
            if "command was not found" in question or "could not find command" in question:
                reflection_content = response_text.lower()
                
                # Look for phrases indicating the main goal was achieved
                for indicator in self.completion_indicators:
                    if indicator in reflection_content:
                        ai_logger.info(f"LLM indicates task complete despite command failures: '{indicator}'")
                        ai_logger.info("Converting awaiting_human_input to task_complete")
                        return {"type": "task_complete"}
        
        return None

    def _normalize_tool_call(self, item: Dict[str, Any]) -> None:
        """Normalize tool call format by standardizing key names."""
        if "type" not in item:
            if "name" in item:
                item["type"] = item.pop("name")
                ai_logger.debug(f"Renamed 'name' key to 'type' for consistent format")
            elif "action" in item:
                item["type"] = item.pop("action")
                ai_logger.debug(f"Renamed 'action' key to 'type' for consistent format")
        
        # Convert legacy execute_commands to pyflipper
        if item.get("type") == "execute_commands":
            item["type"] = "pyflipper"
            ai_logger.debug("Converted 'execute_commands' type to 'pyflipper'")

    def _parse_plan_with_regex(self, response_text: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Use regex to extract JSON from response text for plan parsing."""
        try:
            # Try to find a JSON array first
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                ai_logger.debug(f"Extracted JSON string using regex:\n{json_string}")
                
                plan = json.loads(json_string)
                if isinstance(plan, list):
                    ai_logger.info(f"Successfully parsed plan as JSON array using regex.")
                    
                    for item in plan:
                        self._normalize_tool_call(item)
                        
                        if item.get("type") == "ask_human":
                            ai_logger.info("Converting ask_human type to awaiting_human_input format")
                            return {
                                "type": "awaiting_human_input",
                                "question": item.get("parameters", {}).get("question", "Need more information")
                            }
                    
                    return plan
                else:
                    ai_logger.warning(f"Parsed JSON is not a list: {plan}")
                    return {"type": "invalid_plan", "message": "LLM response was not a JSON array."}

            # Try to find a JSON dictionary
            dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if dict_match:
                json_string = dict_match.group(0)
                ai_logger.debug(f"Extracted JSON dict using regex:\n{json_string}")
                
                parsed_dict = json.loads(json_string)
                if isinstance(parsed_dict, dict):
                    # Check for ask_human variants
                    if parsed_dict.get("type") == "ask_human" or parsed_dict.get("action") == "ask_human":
                        return {
                            "type": "awaiting_human_input",
                            "question": parsed_dict.get("parameters", {}).get("question", "Need more information")
                        }
                    
                    if "type" in parsed_dict:
                        # Handle special types
                        if parsed_dict.get("type") in self.special_response_types:
                            return parsed_dict
                        # Regular tool calls get wrapped
                        ai_logger.info(f"Wrapping single tool call with type '{parsed_dict.get('type')}' in a list (regex extraction)")
                        return [parsed_dict]

            ai_logger.error(f"Could not parse plan or structured response from LLM response:\n{response_text}")
            return {"type": "parsing_failed", "message": "LLM response did not contain a valid plan or structured action."}

        except json.JSONDecodeError as e:
            ai_logger.error(f"JSON Decode Error parsing LLM plan response: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Failed to parse JSON: {e}", "raw_response": response_text}
        except Exception as e:
            ai_logger.error(f"Unexpected error during plan parsing: {e}", exc_info=True)
            return {"type": "parsing_error", "message": f"Unexpected parsing error: {e}"}

    def _parse_reflection_with_regex(self, response_text: str) -> Dict[str, Any]:
        """Use regex to extract JSON from response text for reflection parsing."""
        try:
            # First try to find a JSON dictionary
            dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if dict_match:
                json_string = dict_match.group(0)
                ai_logger.debug(f"Extracted JSON dict using regex:\n{json_string}")

                action = json.loads(json_string)
                if isinstance(action, dict):
                    # If it has a "type" key, treat it as a reflection
                    if "type" in action:
                        if action["type"] in self.valid_reflection_types:
                            ai_logger.info(f"Successfully parsed reflection action from regex: {action['type']}")
                            return action

                    # If it has an "action" key, it might be a tool call to convert
                    if "action" in action:
                        # Convert action to type
                        action["type"] = action.pop("action")
                        ai_logger.debug("Renamed 'action' key to 'type' for consistent format")
                        
                        # Check for special conversions
                        if action.get("type") == "execute_commands":
                            action["type"] = "pyflipper"
                            ai_logger.debug("Converted 'execute_commands' type to 'pyflipper'")

                        if action["type"] == "mark_task_complete":
                            return {"type": "task_complete"}
                        elif action["type"] == "ask_human" or action["type"] == "ask_question":
                            return {
                                "type": "awaiting_human_input",
                                "question": action.get("parameters", {}).get("question", "Need more information")
                            }

            # If no reflection dict found, check for a JSON array
            array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if array_match:
                json_string = array_match.group(0)
                ai_logger.debug(f"Extracted JSON array using regex:\n{json_string}")

                array_data = json.loads(json_string)
                if isinstance(array_data, list) and len(array_data) > 0:
                    ai_logger.info(f"Found a list of {len(array_data)} tool calls - converting to add_tasks")
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