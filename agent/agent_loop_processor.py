"""
Processes user requests and interacts with the Flipper Zero and LLM agents
using the Plan, Act, Reflect loop with UnifiedLLMAgent.
"""

import logging
from typing import Optional, List, Tuple, Dict, Any # Added Dict, Any for task structure

from hardware.hardware_manager import FlipperZeroManager
from .llm_agent import UnifiedLLMAgent
from .agent_loop.agent_state import AgentState # Assuming AgentState is here

# Assuming ToolExecutor and TaskManager might be needed here or imported
# from .agent_loop.task_manager import TaskManager # Example import
# from .agent_loop.tool_executor import ToolExecutor # Example import

logger = logging.getLogger("AgentFlipper")

async def process_user_request_unified(app_instance: Any, user_input: str, flipper_agent: FlipperZeroManager, unified_llm_agent: UnifiedLLMAgent, agent_state: AgentState):
    """
    Process a user request using the Plan, Act, Reflect loop with UnifiedLLMAgent.
    Args:
        app_instance: The TextualApp instance for UI updates. (Assuming this is needed for UI interactions within the loop)
        user_input: The user's input text
        flipper_agent: The agent that communicates with the Flipper Zero (will be used by ToolExecutor)
        unified_llm_agent: The UnifiedLLMAgent instance for LLM interactions (planning, reflection)
        agent_state: The AgentState instance holding the agent's state and context
        # TaskManager will likely be needed here or accessed via agent_state
        # ToolExecutor will also be needed here
    Returns:
        None
    """
    logger.info(f"Processing user request: {user_input} with UnifiedLLMAgent")

    # 1. Receive User Input & Initialize State/Context
    # Assuming AgentState is initialized elsewhere and passed in.
    # Update state with the new user input.
    agent_state.add_user_message(user_input) # Assuming AgentState has this method
    agent_state.clear_task_queue() # Assuming AgentState manages the task queue
    agent_state.reset_loop_count() # Assuming AgentState tracks loop iterations
    agent_state.set_awaiting_human_input(False, "") # Ensure human input flag is off

    # Check device connection before proceeding with LLM calls
    if not flipper_agent.is_connected:
        logger.error("Device is not connected. Attempting reconnection...")
        reconnect_success = flipper_agent.connect()
        if not reconnect_success:
            await app_instance.display_message(f"[bold red]ERROR: Cannot connect to Flipper Zero device. Please check your connection and try again.[/bold red]")
            return # Exit if cannot connect
        else:
            await app_instance.display_message(f"[green]Successfully reconnected to Flipper Zero.[/green]")

    # The core Plan, Act, Reflect loop
    while True:
        # Guard against excessive loop iterations (instead of recursion depth)
        if agent_state.get_loop_count() >= unified_llm_agent.max_recursion_depth: # Assuming UnifiedLLMAgent retains this config
            logger.warning(f"Maximum loop iterations reached ({unified_llm_agent.max_recursion_depth}), stopping task.")
            await app_instance.display_message(f"\nMaximum loop iterations reached ({unified_llm_agent.max_recursion_depth}). Please issue a new command to continue.")
            # Assuming summary generation logic might be handled differently or needed elsewhere
            # For now, explicitly stopping.
            break # Exit the loop

        agent_state.increment_loop_count() # Increment loop counter

        # 3. Plan (Initial Planning) or Act (Executing Tasks)
        if agent_state.is_task_queue_empty() and not agent_state.is_awaiting_human_input():
            # Only plan if the task queue is empty and not waiting for human input
            await app_instance.display_message(f"[purple]Planning...[/purple]")
            # Pass agent_state to create_initial_plan for context if needed
            plan_result = await unified_llm_agent.create_initial_plan(agent_state.get_current_task() or user_input) # Use current task or user input

            # Handle plan_result (list of tasks, awaiting_human_input, etc.)
            logger.debug(f"Initial plan result: {plan_result}")

            if isinstance(plan_result, list):
                # If plan is a list of tasks, add them to the queue
                agent_state.add_plan_to_task_queue(plan_result) # Assuming AgentState has this method
                await app_instance.display_message(f"[green]Plan received. Executing tasks...[/green]")
            elif isinstance(plan_result, dict) and plan_result.get("type") == "awaiting_human_input":
                # If planning requires human input
                question = plan_result.get("question", "Need more information")
                agent_state.set_awaiting_human_input(True, question)
                await app_instance.display_message(f"[yellow]? {question}[/yellow]")
                break # Pause loop for human input
            elif isinstance(plan_result, dict) and plan_result.get("type") == "task_complete":
                 await app_instance.display_message(f"[green]✓ Task completed during planning.[/green]")
                 break # Task complete
            else:
                logger.warning(f"Unexpected plan result format: {plan_result}")
                await app_instance.display_message(f"[orange]Received unexpected response from LLM during planning.[/orange]")
                # Decide how to handle unexpected results - could ask human, or stop. For now, stop.
                break

        # 7. Execute Task (if queue is not empty and not awaiting human input)
        if not agent_state.is_task_queue_empty() and not agent_state.is_awaiting_human_input():
            next_task = agent_state.get_next_task() # Assuming AgentState manages queue and returns next task
            if next_task:
                await app_instance.display_message(f"[purple]Executing Task: {next_task.get('action', 'Unknown')}[/purple]")

                # TODO: Integrate ToolExecutor here
                # The ToolExecutor would take the task (action and parameters)
                # and call the appropriate function (e.g., flipper_agent.execute_commands)
                # This is a placeholder for ToolExecutor execution
                task_result = {"status": "simulated_success", "response": "Simulated result"} # Replace with actual execution
                # Example execution logic for pyflipper:
                if next_task.get("action") == "pyflipper":
                    commands = next_task.get("parameters", {}).get("commands", [])
                    if commands:
                         await app_instance.display_message(f"{'─' * 79}")
                         await app_instance.display_message(f"Executing {len(commands)} commands...")
                         for i, cmd in enumerate(commands, 1):
                             await app_instance.display_message(f"{i}. {cmd}")
                         await app_instance.display_message(f"{'─' * 79}")
                         results = await flipper_agent.execute_commands(commands, app_instance)
                         task_result = {"status": "executed", "results": results}
                         # Add results to state for reflection
                         agent_state.add_task_result(next_task, task_result) # Assuming AgentState tracks task results
                    else:
                        task_result = {"status": "error", "response": "No commands provided for pyflipper"}
                        agent_state.add_task_result(next_task, task_result)

                elif next_task.get("action") == "provide_information":
                    information = next_task.get("parameters", {}).get("information", "No information provided")
                    await app_instance.display_message(f"{'─' * 79}")
                    await app_instance.display_message(f"# Information:")
                    await app_instance.display_message(f"{information}")
                    await app_instance.display_message(f"{'─' * 79}")
                    task_result = {"status": "displayed", "information": information}
                    agent_state.add_task_result(next_task, task_result)

                elif next_task.get("action") == "ask_human":
                     question = next_task.get("parameters", {}).get("question", "Can you provide some input?")
                     agent_state.set_awaiting_human_input(True, question)
                     await app_instance.display_message(f"[yellow]? {question}[/yellow]")
                     task_result = {"status": "awaiting_human_input", "question": question}
                     agent_state.add_task_result(next_task, task_result)
                     break # Pause loop for human input

                elif next_task.get("action") == "mark_task_complete":
                     await app_instance.display_message(f"[green]✓ Task marked complete by LLM.[/green]")
                     task_result = {"type": "task_complete"} # Reflect using the 'type' key
                     agent_state.add_task_result(next_task, task_result)
                     # Continue to reflection - reflection should detect task_complete type and break the loop

                else:
                    logger.warning(f"Unknown tool call: {next_task.get('action')}")
                    await app_instance.display_message(f"[orange]Received unknown tool call: {next_task.get('action')}[/orange]")
                    task_result = {"status": "error", "response": f"Unknown tool: {next_task.get('action')}"}
                    agent_state.add_task_result(next_task, task_result)
                    # Continue to reflection with the error result

                # 9. Reflect
                # Provide the task and its result to the LLM for reflection
                # Assuming agent_state has methods to get context for reflection
                reflection_result = await unified_llm_agent.reflect_and_plan_next(next_task, task_result)
                logger.debug(f"Reflection result: {reflection_result}")

                # 10. Next Step Decision
                if isinstance(reflection_result, dict):
                    if reflection_result.get("type") == "add_tasks":
                        new_tasks = reflection_result.get("tasks", [])
                        agent_state.add_plan_to_task_queue(new_tasks) # Add new tasks to the queue
                        await app_instance.display_message(f"[green]LLM suggested {len(new_tasks)} new tasks.[/green]")
                    elif reflection_result.get("type") == "task_complete":
                        await app_instance.display_message(f"[green]✓ Task completed.[/green]")
                        break # Exit loop
                    elif reflection_result.get("type") == "awaiting_human_input":
                        question = reflection_result.get("question", "Need human input")
                        agent_state.set_awaiting_human_input(True, question)
                        await app_instance.display_message(f"[yellow]? {question}[/yellow]")
                        break # Pause loop for human input
                    elif reflection_result.get("type") == "info":
                         information = reflection_result.get("information", "Information from reflection")
                         await app_instance.display_message(f"{'─' * 79}")
                         await app_instance.display_message(f"# Information (Reflection):")
                         await app_instance.display_message(f"{information}")
                         await app_instance.display_message(f"{'─' * 79}")
                         # Continue loop
                    elif reflection_result.get("type") == "unhandled_reflection" or reflection_result.get("type") == "parsing_error":
                         error_message = reflection_result.get("message", "LLM reflection parsing error")
                         await app_instance.display_message(f"[red]Error processing LLM reflection: {error_message}[/red]")
                         logger.error(f"LLM reflection error: {reflection_result}")
                         break # Stop on reflection errors
                    else:
                        logger.warning(f"Unhandled reflection type: {reflection_result.get('type')}")
                        await app_instance.display_message(f"[orange]Received unhandled reflection type: {reflection_result.get('type')}[/orange]")
                        # Decide how to handle unhandled reflection types - for now, continue if queue not empty, else break.
                        if agent_state.is_task_queue_empty():
                            break # Exit loop if no more tasks and unhandled reflection
                else:
                    logger.warning(f"Unexpected reflection result format: {reflection_result}")
                    await app_instance.display_message(f"[orange]Received unexpected reflection result format.[/orange]")
                    # Decide how to handle unexpected format - for now, continue if queue not empty, else break.
                    if agent_state.is_task_queue_empty():
                        break # Exit loop if no more tasks and unexpected format

            else:
                # Should not happen if is_task_queue_empty() is false, but good practice to check
                logger.warning("Task queue indicated not empty, but get_next_task returned None.")
                break # Exit loop to prevent infinite loop

        elif agent_state.is_awaiting_human_input():
            # Loop should pause when awaiting human input. This break is redundant if the previous break is hit,
            # but added for clarity if loop structure changes.
            break

        else:
            # If queue is empty and not awaiting human input, and not in planning phase, the task might be done or stuck.
            # This could happen if planning resulted in no tasks, or reflection didn't add tasks.
            # If the loop reaches here and queue is empty, it should likely terminate.
             if agent_state.is_task_queue_empty():
                 logger.info("Task queue is empty and not awaiting human input. Exiting loop.")
                 break
             # If not empty, the loop should continue, which the while condition handles.

    # End of the main loop
    logger.info("Exiting process_user_request loop.")
    # Consider final summary generation here if not handled by a task_complete signal