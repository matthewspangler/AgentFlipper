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
        task_description = task.get("description", "Unnamed Task")
        tool_name = task.get("tool_name", "N/A")
        task_id = task.get("id", "N/A")[:4] # Use first 4 chars of ID

        formatted_string = f"{task_description} (Tool: {tool_name}) [ID: {task_id}]"
        return formatted_string