import asyncio
import json # For debugging or displaying complex structures
from typing import Any, Dict

# These will be proper imports once the modules are created
# from .agent_state import AgentState
# from .task_manager import TaskManager
# from .tool_executor import ToolExecutor
# from ..llm.llm_agent import UnifiedLLMAgent
# from ..ui.human_interaction import HumanInteractionHandler
# from ..ui.textual_app import AgentFlipperApp # Or however the app is named

class AgentLoop:
    def __init__(self,
                 agent_state: Any,
                 task_manager: Any,
                 tool_executor: Any,
                 llm_agent: Any,
                 human_interaction_handler: Any,
                 app_instance: Any):
        self.agent_state = agent_state
        self.task_manager = task_manager
        self.tool_executor = tool_executor
        self.llm_agent = llm_agent # This is UnifiedLLMAgent
        self.human_interaction = human_interaction_handler
        self.app_instance = app_instance # Textual App instance

    async def process_user_request(self, user_input: str):
        """Entry point for processing a user request."""
        # Initialize or clear parts of state for a new user request
        self.agent_state.clear_for_new_session() # Resets history, task_queue etc.
        await self.app_instance.display_message(f"Processing: '{user_input}'")

        # The initial user input becomes the first "task" or goal
        self.agent_state.update_context("current_goal", user_input)
        # We don't add user_input directly as a task to TaskManager here,
        # as the first step is LLM planning based on this goal.
        # TaskManager will receive tasks from the planner.

        # Update UI to show initial state (empty task list)
        await self.app_instance.update_task_list_ui() # Added call here

        # Start the main loop
        await self._run_main_loop()

    async def _run_main_loop(self):
        """Main agent loop as depicted in the diagram."""
        loop_iteration = 0 # For safety break if needed

        while True:
            loop_iteration += 1
            if loop_iteration > self.agent_state.config.get("max_loop_iterations", 20): # Safety break
                await self.app_instance.display_message("⚠️ Max loop iterations reached. Ending task.")
                break

            # Check if we're waiting for human input
            if self.agent_state.awaiting_human_input:
                # The UI handles getting input and updating agent_state.
                # AgentLoop just yields control if waiting.
                # If UI uses an event to signal input, AgentLoop could await that event.
                # For simplicity, we'll rely on the UI clearing `awaiting_human_input`.
                await asyncio.sleep(0.1)  # Short sleep to avoid CPU spinning

                # After human input, the LLM might need to process it.
                # This is handled by UnifiedLLMAgent.process_human_input
                # which might be called from the UI callback or here.
                # For now, assume UI has updated state, and loop continues.
                # If human input led to new tasks, they should be in the queue.
                if not self.agent_state.awaiting_human_input: # If human input was processed
                    # Human input might generate a new plan, let's try to get it
                    human_response_context = self.agent_state.task_results.get("human_input_processed") # Example
                    if human_response_context:
                         new_plan = await self.llm_agent.process_human_input(
                             self.agent_state.context_buffer.get("current_goal"), # Or more specific context
                             human_response_context.get("response")
                         )
                         if new_plan and isinstance(new_plan, list):
                            self.task_manager.add_plan_to_queue(new_plan)
                            await self.app_instance.update_task_list_ui() # Added call here (after adding new plan from human input)
                         elif new_plan and new_plan.get("type") == "awaiting_human_input":
                            # LLM asked for more human input after human input, allow this
                            pass # State is already set
                continue # Re-evaluate loop conditions

            # Check if any tasks remain in the queue
            if self.task_manager.is_empty():
                current_goal = self.agent_state.context_buffer.get("current_goal")
                if not current_goal: # No goal means task is done or never started
                    await self.app_instance.display_message("🏁 No current goal. Task considered complete.")
                    break

                # Generate initial plan or next part of plan from LLM
                await self.app_instance.display_message("🧠 Thinking... (Requesting plan)")
                await self.app_instance.display_message(f"Creating plan for goal: '{current_goal}'")
                plan = await self.llm_agent.create_initial_plan(current_goal) # Or get_next_plan
                await self.app_instance.display_message(f"Plan received from LLM: {json.dumps(plan, indent=2)}")

                # Handle the plan returned by LLM. It can be a list of actions or a signal dictionary.
                if isinstance(plan, dict) and plan.get("type") == "awaiting_human_input":
                    # LLM decided it needs human input to plan. State is set by llm_agent.
                    # Loop will pick up `awaiting_human_input` on next iteration.
                    await self.app_instance.display_message("👤 LLM is requesting human input to proceed with planning.")
                    continue

                elif isinstance(plan, list) and plan: # Check if it's a non-empty list (a valid plan)
                    # Request human approval for the plan if configured
                    if self.agent_state.config.get("require_plan_approval", False):
                        await self.human_interaction.handle_approval_request(plan)
                        # This sets awaiting_human_input. Loop will pause.
                        # The UI will update task list when approval is processed.
                        continue

                    # If approved or no approval required, add plan to queue
                    await self.app_instance.display_message(f"📝 Plan received. Adding {len(plan)} steps to task queue.")
                    self.task_manager.add_plan_to_queue(plan)
                    await self.app_instance.update_task_list_ui() # Added call here (after adding initial plan)

                    # Continue loop to process the newly added tasks
                    continue # Go to the next loop iteration which will get the first task from the queue

                else: # Plan is invalid (None, empty list, or unexpected format)
                    await self._handle_invalid_plan(plan)
                    # If plan is invalid, maybe ask human or terminate
                    if self.agent_state.config.get("human_error_resolution", False):
                         await self.human_interaction.handle_error_resolution("Failed to generate or parse a valid plan.", {"goal": current_goal, "raw_llm_response": plan}) # Pass the invalid plan/response for context
                         continue # Loop will pick up awaiting_human_input
                    else:
                        await self.app_instance.display_message("❌ Failed to generate or parse a valid plan. Ending task.")
                        break # End task if planning fails and no HITL for it


            # If task queue is not empty (either had tasks initially or a valid plan was added)
            if not self.task_manager.is_empty():
                 # Get next task from queue
                 task = self.task_manager.get_next_task()
                 await self.app_instance.update_task_list_ui() # Added call here (after getting next task)
                 if not task: # Should not happen if is_empty() was false, but defensive check
                    continue
            else: # Task queue is empty and planning did not add tasks (or planning failed and broke loop)
                # This could mean the LLM decided the goal is met implicitly, or there's nothing left.
                await self.app_instance.display_message("🏁 Task queue empty. Assuming task complete.")
                break
            if not task: # Should not happen if is_empty() was false, but defensive
                continue

            await self.app_instance.display_message(f"▶️ Executing: {task.get('action')}...")

            # Execute the task
            result = await self.tool_executor.execute_task(task)

            # Store result in state
            self.agent_state.task_results[task.get("id", "unknown")] = result
            self.agent_state.add_to_history({"role": "assistant", "content": f"Action: {task.get('action')}, Result: {json.dumps(result)}"})

            # If task execution failed, consider asking for human help
            if not result.get("success", False):
                if self.agent_state.config.get("human_error_resolution", False):
                    await self.human_interaction.handle_error_resolution(
                        result.get("error", "Unknown error"),
                        task
                    )
                    continue # Loop will pick up awaiting_human_input
                else:
                    # If no HITL for errors, log and maybe LLM reflection can handle it.
                    await self.app_instance.display_message(f"⚠️ Error executing {task.get('action')}: {result.get('error')}")


            # Give result to LLM for reflection and next steps
            await self.app_instance.display_message("🤔 Reflecting on results...")
            reflection_action = await self.llm_agent.reflect_and_plan_next(task, result)

            if reflection_action and reflection_action.get("type") == "awaiting_human_input":
                # LLM reflection decided it needs human input. State is set by llm_agent.
                continue

            # Handle reflection outcome
            if reflection_action and reflection_action.get("type") == "add_tasks":
                new_tasks = reflection_action.get("tasks", [])
                if new_tasks:
                    await self.app_instance.display_message(f"➕ Adding {len(new_tasks)} new sub-task(s) to queue.")
                    self.task_manager.add_plan_to_queue(new_tasks) # Assuming plan format
                    await self.app_instance.update_task_list_ui() # Added call here (after adding tasks from reflection)
            elif reflection_action and reflection_action.get("type") == "task_complete":
                await self.app_instance.display_message("✅ LLM signaled task complete.")
                # Optionally, could ask for final human verification here if configured
                if self.agent_state.config.get("human_final_verification", False):
                    await self.human_interaction.request_input("verification", "Task marked complete. Is this correct?", ["Yes", "No"])
                    continue # Loop will pause for verification
                break # Exit criteria met
            else: # No new tasks, no completion signal, maybe LLM provided info or an error
                # This part needs careful handling. If LLM just gives info, display it.
                # If it's an unhandled state, could be an issue.
                # For now, if no clear next step or completion, we might break or ask human.
                if self.agent_state.config.get("human_on_ambiguity", True) and not self.task_manager.is_empty():
                    # If queue has items, we might proceed, but if LLM gave no direction, it's ambiguous
                     await self.human_interaction.request_input("clarification", "LLM reflection did not provide a clear next step. How to proceed?")
                     continue
                elif not self.task_manager.is_empty():
                    # Continue processing queue if LLM reflection was just informational
                    pass
                else:
                    await self.app_instance.display_message("🏁 No further actions planned by LLM. Task considered complete.")
                    break


        # End of processing
        await self.app_instance.display_message("Processing complete for this request.")
        await self.app_instance.update_task_list_ui() # Added call here (at the very end)

    def _initialize_context_buffer(self):
        """Set up or update the context buffer based on current goal."""
        # Example: self.agent_state.context_buffer["some_key"] = "some_value"
        # The current_goal is set in process_user_request
        pass

    async def _handle_invalid_plan(self, plan):
        """Handle case when LLM produces invalid plan."""
        # Log the invalid plan for debugging
        print(f"Invalid plan received: {plan}") # Replace with proper logging
        await self.app_instance.display_message("⚠️ LLM generated an invalid plan.")