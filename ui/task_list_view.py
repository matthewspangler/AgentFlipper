"""
Textual widget to display the agent's task list in a tree view.
"""

from textual.widgets import Tree
from textual.app import ComposeResult
from textual.containers import Container
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from ..agent_loop.agent_state import AgentState

class TaskListTreeView(Tree[None]):
    """A Tree widget to display the agent's task queue."""

    def __init__(self, agent_state: "AgentState", **kwargs):
        """Initialize the TaskListTreeView."""
        super().__init__("Tasks", **kwargs)
        self.agent_state = agent_state
        self.show_root = False  # Don't show the "Tasks" root node initially

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_task_display()
        # TODO: Set up a mechanism to call update_task_display when agent_state.task_queue or current_task changes

    def update_task_display(self) -> None:
        """Clears and repopulates the tree view based on agent_state."""
        self.clear() # Clear existing nodes

        if self.agent_state.current_task:
            current_task_node = self.root.add(f"[b]⏳ {self._format_task(self.agent_state.current_task)}[/b]")
            # Optionally add task details as children of the current task node
            # current_task_node.add("Status: In Progress")
            # current_task_node.add(f"Tool: {self.agent_state.current_task.get('tool_name', 'N/A')}")

        if self.agent_state.task_queue:
            self.root.add("[b]Queued Tasks:[/b]")
            for task in self.agent_state.task_queue:
                self.root.add(f"- {self._format_task(task)}")
        elif not self.agent_state.current_task:
             self.root.add("[italic]No tasks in queue.[/italic]")


    def _format_task(self, task: Dict[str, Any]) -> str:
        """Formats a task dictionary into a human-readable string."""
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "N/A")[:4] # Use first 4 chars of ID

        if task_type == "user_request":
            # For the initial user request task
            task_description = f"Goal: {task.get('content', 'Unnamed Request')}"
            tool_info = "N/A" # User request doesn't have a tool
        elif task_type == "planned_action":
            # For tasks generated from the LLM's plan
            tool_name = task.get("action", "N/A")
            # Try to get a concise description from parameters or just use tool name
            parameters = task.get("parameters", {})
            if tool_name == "pyflipper" and "commands" in parameters and isinstance(parameters["commands"], list):
                 # Display the first command or a summary
                 commands = parameters["commands"]
                 if commands:
                     task_description = f"Execute: {commands[0][:30]}..." if len(commands[0]) > 30 else f"Execute: {commands[0]}"
                 else:
                     task_description = "pyflipper Commands (no commands)"
                 tool_info = tool_name
            elif tool_name == "provide_information" and "information" in parameters:
                 task_description = f"Info: {parameters['information'][:30]}..." if len(parameters['information']) > 30 else f"Info: {parameters['information']}"
                 tool_info = tool_name
            elif tool_name == "ask_question" and "question" in parameters:
                 task_description = f"Ask: {parameters['question'][:30]}..." if len(parameters['question']) > 30 else f"Ask: {parameters['question']}"
                 tool_info = tool_name
            else:
                 # Default description for other planned actions
                 task_description = f"Action: {tool_name}"
                 tool_info = tool_name
        else:
            # For unknown task types
            task_description = f"Unknown Task Type ({task_type})"
            tool_info = "N/A"

        formatted_string = f"{task_description} (Tool: {tool_info}) [ID: {task_id}]"
        return formatted_string