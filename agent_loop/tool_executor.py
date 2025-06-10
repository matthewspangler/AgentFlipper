import json
from typing import Dict, Any, Callable, Awaitable

# Assuming AgentState is in the same package or path is adjusted
# from .agent_state import AgentState
# Assuming Textual App instance for UI updates
# from ..ui.textual_app import AgentFlipperApp # Or however the app is named/imported

class ToolExecutor:
    def __init__(self, agent_state: Any, app_instance: Any): # Use Any for now
        self.agent_state = agent_state
        self.app_instance = app_instance # This is the Textual App instance
        self.tools: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = self._register_tools()
        
    def _register_tools(self) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """Register all available tools with their execution functions."""
        return {
            "execute_commands": self._execute_flipper_commands,
            "provide_information": self._provide_information,
            "ask_question": self._ask_question, # This tool signals HITL
            # Add other tools as needed
        }
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on its type and action."""
        tool_name = task.get("action")
        parameters = task.get("parameters", {})
        task_id = task.get("id", "unknown_task")

        try:
            if tool_name not in self.tools:
                # Log this attempt to call an unknown tool
                print(f"Error: Unknown tool '{tool_name}' for task ID {task_id}") # Replace with proper logging
                raise ValueError(f"Unknown tool: {tool_name}")
                
            # Execute the appropriate tool function
            tool_function = self.tools[tool_name]
            result_payload = await tool_function(parameters)
            
            # Standardize result format
            return {
                "success": True,
                "task_id": task_id,
                "result": result_payload # The direct output from the tool function
            }
            
        except Exception as e:
            # Log the full error
            print(f"Error executing tool '{tool_name}' for task ID {task_id}: {e}") # Replace with proper logging
            # Standardized error handling
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
    async def _execute_flipper_commands(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute commands on Flipper device."""
        commands = parameters.get("commands", [])
        
        # Improved logging
        await self.app_instance.display_message(f"📤 Executing commands: {commands}")
        
        if not isinstance(commands, list):
            await self.app_instance.display_message(f"⚠️ Error: Commands parameter must be a list")
            return {"error": "Commands parameter must be a list."}
        
        # Ensure flipper_agent is available via agent_state
        if not self.agent_state.flipper_agent:
            await self.app_instance.display_message(f"⚠️ Error: Flipper agent not initialized")
            return {"error": "Flipper agent not initialized."}

        try:
            # Assuming execute_commands returns a list of (command, response) tuples
            results = await self.agent_state.flipper_agent.execute_commands(commands, self.app_instance)
            await self.app_instance.display_message(f"📥 Command results: {json.dumps(results, indent=2)}")
            return {"executed_commands": results, "success": True}
        except Exception as e:
            error_msg = f"Error executing commands: {str(e)}"
            await self.app_instance.display_message(f"⚠️ {error_msg}")
            return {"error": error_msg, "success": False}
        
    async def _provide_information(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Display information to the user."""
        information = parameters.get("information", "")
        await self.app_instance.display_message(f"ℹ️ {information}") # Uses app_instance directly
        return {"displayed": True, "information": information}
        
    async def _ask_question(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signals that a question needs to be asked to the human.
        This tool's execution essentially triggers the HITL flow.
        The actual asking and waiting is handled by HumanInteractionHandler & AgentLoop.
        """
        question = parameters.get("question", "I need more information.")
        # This tool's "result" indicates that a human interaction is now required.
        # The HumanInteractionHandler will use this question.
        return {"type": "ask_human", "question": question}