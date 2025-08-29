import asyncio
import json
import logging
from typing import Any, Dict

from .agent_state import AgentState
from .task_manager import TaskManager
from .tool_executor import ToolExecutor
from .llm_agent import UnifiedLLMAgent
from ui.human_interaction import HumanInteractionHandler

class AgentLoop:
    def __init__(self,
                 agent_state: AgentState,
                 task_manager: TaskManager,
                 tool_executor: ToolExecutor,
                 llm_agent: UnifiedLLMAgent,
                 human_interaction_handler: HumanInteractionHandler,
                 app_instance: Any):
        self.agent_state = agent_state
        self.task_manager = task_manager
        self.tool_executor = tool_executor
        self.llm_agent = llm_agent # This is UnifiedLLMAgent
        self.human_interaction = human_interaction_handler
        self.app_instance = app_instance # Textual App instance

    async def process_user_request(self, user_input: str):
        """Entry point for processing a user request."""
        self.agent_state.clear_for_new_session()

        self.agent_state.update_context("current_goal", user_input)

        # Update UI to show initial state (empty task list)
        await self.app_instance.update_task_list_ui()

        # Check device connection before proceeding
        if not self.agent_state.flipper_agent or not self.agent_state.flipper_agent.is_connected:
            await self.app_instance.display_message("[bold red]ERROR: Flipper Zero not connected.[/bold red]")
            return

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

            if self.agent_state.awaiting_human_input:
                # AgentLoop yields control while waiting for human input.
                # The UI is responsible for getting input and updating agent_state,
                # which will clear `awaiting_human_input`.
                await asyncio.sleep(0.1) # Short sleep to avoid high CPU usage

                # If human input was just processed (awaiting_human_input is now False)
                if not self.agent_state.awaiting_human_input:
                     # The UI or HumanInteractionHandler might have added new tasks
                     # based on human input. The loop will proceed to check the queue.
                     # If human input requires LLM processing to generate next steps,
                     # that would happen here or in the UI/HumanInteractionHandler after input is received.
                     # For now, assuming human input directly affects state or queue.
                     pass # Continue loop to check queue

                continue # Re-evaluate loop conditions

            # Check if any tasks remain in the queue
            if self.task_manager.is_empty():
                current_goal = self.agent_state.context_buffer.get("current_goal")
                if not current_goal:
                    await self.app_instance.display_message("🏁 No current goal. Task considered complete.")
                    break

                # Generate initial plan or next part of plan from LLM
                await self.app_instance.display_message("🧠 Thinking... (Requesting plan)")
                plan = await self.llm_agent.create_initial_plan(current_goal)


                # Handle the plan returned by LLM. It can be a list of tool calls or a signal dictionary.
                if isinstance(plan, dict) and plan.get("type") == "awaiting_human_input":
                    # LLM decided it needs human input to plan. State is set by llm_agent.
                    await self.app_instance.display_message("👤 LLM is requesting human input to proceed with planning.")
                    continue

                elif isinstance(plan, list) and plan:
                    # Request human approval for the plan if configured
                    if self.agent_state.config.get("require_plan_approval", False):
                        await self.human_interaction.handle_approval_request(plan)
                        continue # Loop will pause for approval

                    # If approved or no approval required, add plan to queue
                    await self.app_instance.display_message(f"[#ff9722]{'─' * 79}[/#ff9722]")
                    await self.app_instance.display_message(f"📝 Plan received. Adding {len(plan)} steps to task queue.")
                    for i, step in enumerate(plan, 1):
                        await self.app_instance.display_message(f"{i}. {step.get('type')}: {step.get('parameters')}")
                    await self.app_instance.display_message(f"[#ff9722]{'─' * 79}[/#ff9722]")
                    self.task_manager.add_plan_to_queue(plan)
                    await self.app_instance.update_task_list_ui()

                    continue # Go to the next loop iteration to process new tasks

                else: # Plan is invalid
                    await self._handle_invalid_plan(plan)
                    if self.agent_state.config.get("human_error_resolution", False):
                         # Ask human for help resolving the invalid plan
                         await self.human_interaction.handle_error_resolution("LLM failed to generate a valid plan.", {"goal": current_goal, "raw_llm_response": plan})
                         continue # Loop will pick up awaiting_human_input
                    else:
                        await self.app_instance.display_message("❌ Failed to generate a valid plan. Ending task.")
                        break # End task


            # If task queue is not empty
            task = self.task_manager.get_next_task()
            if not task: # Should not happen if is_empty() was false, but defensive check
               await self.app_instance.display_message("🏁 Task queue unexpectedly empty. Ending task.")
               break

            await self.app_instance.update_task_list_ui() # Update UI after getting task
            await self.app_instance.display_message(f"▶️  Executing: {task.get('type')}...")

            # Execute the task
            result = await self.tool_executor.execute_task(task)

            # Store result in state
            self.agent_state.task_results[task.get("id", "unknown")] = result
            # Adding result to history for LLM reflection
            self.agent_state.add_to_history({"role": "system", "content": f"Task Result: {json.dumps(result)}"})


            # If task execution failed, consider asking for human help
            if not result.get("success", False):
                await self.app_instance.display_message(f"⚠️ Error executing {task.get('type')}: {result.get('error')}")
                if self.agent_state.config.get("human_error_resolution", False):
                    await self.human_interaction.handle_error_resolution(
                        result.get("error", "Unknown error"),
                        task
                    )
                    continue # Loop will pick up awaiting_human_input
                # If no HITL for errors, LLM reflection can handle it below.


            # Give result to LLM for reflection and next steps
            await self.app_instance.display_message("🤔 Reflecting on results...")
            reflection_action = await self.llm_agent.reflect_and_plan_next(task, result)
            await self.app_instance.display_message(f"Reflection action: {json.dumps(reflection_action)}") # Added display

            if reflection_action and reflection_action.get("type") == "awaiting_human_input":
                # LLM reflection decided it needs human input. State is set by llm_agent.
                continue

            # Handle reflection outcome
            if reflection_action and reflection_action.get("type") == "add_tasks":
                new_tasks = reflection_action.get("tasks", [])
                if new_tasks:
                    await self.app_instance.display_message(f"➕ Adding {len(new_tasks)} new sub-task(s) to queue from reflection.")
                    self.task_manager.add_plan_to_queue(new_tasks)
                    await self.app_instance.update_task_list_ui()
            elif reflection_action and reflection_action.get("type") == "task_complete":
                await self.app_instance.display_message("✅ LLM signaled task complete.")
                if self.agent_state.config.get("human_final_verification", False):
                    await self.human_interaction.request_input("verification", "Task marked complete. Is this correct?", ["Yes", "No"])
                    continue # Loop will pause for verification
                break # Exit criteria met
            elif reflection_action and reflection_action.get("type") == "info":
                 await self.app_instance.display_message(f"ℹ️ LLM Info: {reflection_action.get('information')}")
            # If reflection_action is None or unhandled type, the loop continues to check the queue.
            # If the queue is empty after this, the "if self.task_manager.is_empty():" block
            # at the top of the loop will handle ending the task or replanning.


        # End of processing
        await self.app_instance.display_message("Processing complete for this request.")
        await self.app_instance.update_task_list_ui()

    async def _handle_invalid_plan(self, plan):
        """Handle case when LLM produces invalid plan."""
        # Use logging instead of print
        log = logging.getLogger(__name__)
        log.error(f"Invalid plan received from LLM: {plan}")
        await self.app_instance.display_message("⚠️ LLM generated an invalid plan. Check logs for details.")