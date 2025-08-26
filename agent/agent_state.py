import time
import uuid # For generating unique IDs if needed for tasks

# Assuming AgentState is in the same package or path is adjusted
# from .agent_loop.agent_state import AgentState
# For now, we'll assume AgentState is passed and has task_queue attribute

class AgentState:
    def __init__(self, config=None):
        self.conversation_history = []
        self.context_buffer = {}
        self.task_queue = []
        self.current_task = None
        self.task_results = {}
        self.config = config or {}
        
        self.flipper_agent = None
        self.llm_agent = None
        self.app_instance = None

        self.awaiting_human_input = False
        self.human_input_request = None # Stores details of the human input request
        
    def add_to_history(self, entry):
        """Add entry to conversation history with optional pruning."""
        # Basic implementation, pruning can be added later
        self.conversation_history.append(entry)
        # Example pruning: Keep last N messages
        # MAX_HISTORY_LEN = self.config.get("max_history_length", 50)
        # if len(self.conversation_history) > MAX_HISTORY_LEN:
        #     self.conversation_history = self.conversation_history[-MAX_HISTORY_LEN:]
        
    def update_context(self, key, value):
        """Update a value in the context buffer."""
        self.context_buffer[key] = value
        
    def clear_for_new_session(self):
        """Reset state for a new user session/request."""
        self.conversation_history = []
        # self.context_buffer = {} # Decide if context buffer should be fully cleared
        self.task_queue = []
        self.current_task = None
        self.task_results = {}
        self.awaiting_human_input = False
        self.human_input_request = None
        
    def request_human_input(self, request_type, prompt, options=None):
        """Set state to await human input with given prompt."""
        self.awaiting_human_input = True
        self.human_input_request = {
            "type": request_type,  # "question", "approval", "selection", etc.
            "prompt": prompt,
            "options": options,    # Optional list of choices
            "timestamp": time.time()
        }
        
    def process_human_input(self, input_value):
        """Process received human input and update state."""
        if not self.awaiting_human_input or self.human_input_request is None:
            # Log or handle error: received human input when not expecting it
            return None 

        self.awaiting_human_input = False
        
        history_entry = {
            "role": "human",
            "content": f"Response to '{self.human_input_request['type']}': {input_value}"
        }
        self.add_to_history(history_entry)
        
        result = {"request": self.human_input_request, "response": input_value}
        self.human_input_request = None # Clear the request
        return result

    def get_full_context_for_llm(self):
        """Prepares the full context including history and current state for LLM."""
        context = self.conversation_history.copy()
        if self.current_task:
            context.append({"role": "system", "content": f"Current task: {self.current_task}"})
        if self.task_queue:
            task_list_str = []
            for task in self.task_queue[:3]:
                task_list_str.append(f"- {task.get('action')}: {task.get('parameters', {})}")
            
            if len(self.task_queue) > 3:
                task_list_str.append(f"...and {len(self.task_queue) - 3} more task(s).")
                
            context.append({"role": "system", "content": f"Remaining tasks in queue:\n" + "\n".join(task_list_str)})
        return context