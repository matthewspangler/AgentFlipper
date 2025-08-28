import time
import uuid
from typing import List, Dict, Any, Optional

# Assuming AgentState is in the same package or path is adjusted
# from .agent_state import AgentState
# For now, we'll assume AgentState is passed and has task_queue attribute

# from .agent_state import AgentState # Assuming AgentState is in the same package

class TaskManager:
    def __init__(self, agent_state: Any):
        self.agent_state = agent_state
        
    def add_user_task(self, user_input: str) -> None:
        task = self._create_task_from_user_input(user_input)
        self.add_task(task)
        
    def add_plan_to_queue(self, plan: Optional[List[Dict[str, Any]]]) -> None:
        if not plan:
            return
        for task_data in plan:
            if isinstance(task_data, dict):
                task = self._create_task_from_plan_item(task_data)
                self.add_task(task)
            # TODO: Log or handle malformed plan item
            
    def add_task(self, task: Dict[str, Any]) -> None:
        self.agent_state.task_queue.append(task)
        
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        if self.is_empty():
            return None
        self.agent_state.current_task = self.agent_state.task_queue.pop(0)
        return self.agent_state.current_task
        
    def is_empty(self) -> bool:
        return len(self.agent_state.task_queue) == 0
        
    def _create_task_from_user_input(self, user_input: str) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "source_type": "user_request",
            "content": user_input,
            "status": "pending",
            "created_at": time.time(),
            "sub_tasks": [] # TODO: Implement support for sub-tasks if user request implies sub-goals
        }
        
    def _create_task_from_plan_item(self, plan_item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "source_type": "planned_action",
            "type": plan_item.get("type"),
            "parameters": plan_item.get("parameters", {}),
            "status": "pending",
            "created_at": time.time(),
            "result": None
        }