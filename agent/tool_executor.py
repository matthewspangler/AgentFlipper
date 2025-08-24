import json
from typing import Dict, Any, Callable, Awaitable

# from .agent_state import AgentState # Assuming AgentState is in the same package
# from ..ui.textual_app import AgentFlipperApp # Assuming Textual App instance for UI updates

class ToolExecutor:
    def __init__(self, agent_state: Any, app_instance: Any):
        self.agent_state = agent_state
        self.app_instance = app_instance
        self.tools: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = self._register_tools()
        
    def _register_tools(self) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """Register all available tools with their execution functions."""
        return {
            "pyflipper": self._execute_flipper_commands,
            "provide_information": self._provide_information,
            "ask_question": self._ask_question,
            # TODO: Add other tools as they are implemented
        }
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on its type and action."""
        tool_name = task.get("action")
        parameters = task.get("parameters", {})
        task_id = task.get("id", "unknown_task")

        try:
            if tool_name not in self.tools:
                log = logging.getLogger(__name__)
                log.error(f"Attempted to call unknown tool '{tool_name}' for task ID {task_id}")
                raise ValueError(f"Unknown tool: {tool_name}")
                
            tool_function = self.tools[tool_name]
            result_payload = await tool_function(parameters)
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result_payload
            }
            
        except Exception as e:
            log = logging.getLogger(__name__)
            log.error(f"Error executing tool '{tool_name}' for task ID {task_id}: {e}", exc_info=True)
            
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
    async def _execute_flipper_commands(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute commands on Flipper device."""
        commands = parameters.get("commands", [])
        
        # Use logging for internal details, UI for user-facing info
        log = logging.getLogger(__name__)
        log.info(f"Received request to execute commands: {commands}")
        
        if not isinstance(commands, list):
            log.error(f"Invalid commands parameter: {commands}. Expected a list.")
            await self.app_instance.display_message(f"⚠️ Error: Commands parameter must be a list")
            return {"error": "Commands parameter must be a list."}
        
        if not self.agent_state.flipper_agent:
            log.error("Flipper agent not initialized in AgentState.")
            await self.app_instance.display_message(f"⚠️ Error: Flipper agent not initialized")
            return {"error": "Flipper agent not initialized."}
 
        try:
            results = await self.agent_state.flipper_agent.execute_commands(commands, self.app_instance)
            log.info(f"Finished executing commands. Results: {results}")
            return {"executed_commands": results, "success": True}
        except Exception as e:
            log.error(f"Exception during command execution: {str(e)}", exc_info=True)
            error_msg = f"Error executing commands: {str(e)}"
            # Error message is already displayed by execute_commands, no need to duplicate here.
            return {"error": error_msg, "success": False}
        
    async def _provide_information(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Display information to the user."""
        information = parameters.get("information", "")
        log = logging.getLogger(__name__)
        log.info(f"Displaying information to user: {information}")
        
        await self.app_instance.display_message(f"ℹ️ {information}")
        return {"displayed": True, "information": information}
        
    async def _ask_question(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signals that a question needs to be asked to the human.
        This tool's execution essentially triggers the HITL flow managed by AgentLoop/HumanInteractionHandler.
        The actual asking and waiting is handled outside this executor.
        """
        question = parameters.get("question", "I need more information.")
        log = logging.getLogger(__name__)
        log.info(f"Asked human question via tool: {question}")

        # The result indicates to the AgentLoop that human input is needed.
        # The HumanInteractionHandler will pick up the question from AgentState.
        # Returning a structured dict for the AgentLoop to process.
        return {"type": "awaiting_human_input", "question": question}