import time
import uuid
from typing import List, Dict, Any, Optional

# Assuming AgentState is in the same package or path is adjusted
# from .agent_state import AgentState 
# For now, we'll assume AgentState is passed and has task_queue attribute

class TaskManager:
    def __init__(self, agent_state: Any): # Use Any for now to avoid circular deps if AgentState imports this
        self.agent_state = agent_state
        
    def add_user_task(self, user_input: str) -> None:
        """Convert user input to a task and add to queue."""
        task = self._create_task_from_user_input(user_input)
        self.add_task(task)
        
    def add_plan_to_queue(self, plan: Optional[List[Dict[str, Any]]]) -> None:
        """Add a sequence of tasks from a plan to the queue."""
        if not plan: # Handle cases where plan might be None or empty
            return
        for task_data in plan:
            if isinstance(task_data, dict): # Ensure task_data is a dictionary
                task = self._create_task_from_plan_item(task_data)
                self.add_task(task)
            # Else: log or handle malformed plan item
            
    def add_task(self, task: Dict[str, Any]) -> None:
        """Add a single task to the queue."""
        self.agent_state.task_queue.append(task) # task_queue is on AgentState
        
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get and remove the next task from the queue."""
        if self.is_empty():
            return None
        # Use pop(0) for FIFO queue behavior
        self.agent_state.current_task = self.agent_state.task_queue.pop(0)
        return self.agent_state.current_task
        
    def is_empty(self) -> bool:
        """Check if task queue is empty."""
        return len(self.agent_state.task_queue) == 0
        
    def _create_task_from_user_input(self, user_input: str) -> Dict[str, Any]:
        """Helper to create proper task structure from user input."""
        return {
            "id": str(uuid.uuid4()),
            "type": "user_request", # This task represents the initial user goal
            "content": user_input,
            "status": "pending",
            "created_at": time.time(),
            "sub_tasks": [] # For potential future use if user request implies sub-goals
        }
        
    def _create_task_from_plan_item(self, plan_item: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to create proper task structure from a plan item (tool call)."""
        return {
            "id": str(uuid.uuid4()),
            "type": "planned_action", # This task is a specific action from the LLM's plan
            "action": plan_item.get("action"), # Name of the tool/function to call
            "parameters": plan_item.get("parameters", {}), # Arguments for the tool
            "status": "pending",
            "created_at": time.time(),
            "result": None
        }