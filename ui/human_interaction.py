import json
from typing import Union, Dict, Any, Optional, List

# Assuming AgentState is available
# from ..agent_loop.agent_state import AgentState
# Assuming Textual App instance for UI updates
# from .textual_app import AgentFlipperApp # Or however the app is named/imported

class HumanInteractionHandler:
    def __init__(self, agent_state: Any, app_instance: Any): # Use Any for now
        self.agent_state = agent_state
        self.app_instance = app_instance # This is the Textual App instance
        
    async def request_input(self, request_type: str, prompt: str, options: Optional[List[str]] = None) -> None:
        """
        Requests input from the human by updating the agent state and signaling the UI.
        Does NOT wait for input; the AgentLoop handles waiting.
        """
        # Update agent state to reflect that we are awaiting human input
        self.agent_state.request_human_input(request_type, prompt, options)
        
        # Signal the UI to display the request
        # Assuming app_instance has a method to handle this
        await self.app_instance.display_human_request(
            request_type, 
            prompt, 
            options
        )
        
        # The AgentLoop will see agent_state.awaiting_human_input is True and pause itself.
        # The UI's input handling will need to detect when input is received
        # and update agent_state.awaiting_human_input = False
        # and potentially notify the AgentLoop to resume.

    async def handle_approval_request(self, plan: List[Dict[str, Any]]) -> None:
        """Requests approval for a plan before execution."""
        plan_display = self._format_plan_for_display(plan)
        prompt = f"Would you like to approve this plan?\n\n{plan_display}"
        options = ["Approve", "Reject", "Modify"] # Standard options for approval
        
        await self.request_input("approval", prompt, options)
        
    async def handle_clarification(self, question: str) -> None:
        """Requests clarification from the human."""
        await self.request_input("question", question)
        
    async def handle_error_resolution(self, error: Union[str, Dict[str, Any]], task: Dict[str, Any]) -> None:
        """Requests human help for error resolution."""
        # Format error and task context for the human
        error_display = str(error) if isinstance(error, str) else json.dumps(error, indent=2)
        task_display = json.dumps(task, indent=2)

        prompt = f"An error occurred during task execution:\n{error_display}\n\nTask details:\n{task_display}\n\nHow would you like to proceed?"
        options = ["Retry Task", "Skip Task", "Abort Task", "Provide Alternative Commands"] # Options for error resolution
        
        await self.request_input("error_resolution", prompt, options)
        
    def _format_plan_for_display(self, plan: List[Dict[str, Any]]) -> str:
        """Formats a plan nicely for human display."""
        if not plan:
            return "Empty plan."

        result = "Proposed Plan:\n"
        for i, step in enumerate(plan, 1):
            action = step.get("action", "Unknown Action")
            parameters = step.get("parameters", {})
            result += f"{i}. {action}"
            if parameters:
                # Display parameters concisely
                params_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
                result += f" ({params_str})"
            result += "\n"
        return result