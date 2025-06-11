"""
Handles user interface interactions using Textualize/textual.
"""

import time
import sys
import logging
import asyncio
from functools import partial
import re # Import re for regex
from typing import Dict, Any, Optional, List, Tuple, Callable

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Tree # Added Tree for TaskListTreeView
from textual.containers import Container
from textual.reactive import var
from textual.worker import Worker, get_current_worker
import asyncio # Import asyncio
from ui.task_list_view import TaskListTreeView # Import the new task list view

# Placeholder imports for now to avoid circular dependencies and enable initial structure
AgentLoop = Any
AgentState = Any
HumanInteractionHandler = Any
UnifiedLLMAgent = Any # Still needed for type hinting in on_mount for clarity
TaskManager = Any # Still needed for type hinting in on_mount for clarity
ToolExecutor = Any # Still needed for type hinting in on_mount for clarity

import logging
logger = logging.getLogger("AgentFlipper")

# Remove the old process_request_in_worker function as its logic is replaced by AgentLoop
# async def process_request_in_worker(...) -> None: ... # Function removed


class AgentFlipper(App):
    """Textual application for the AgentFlipper."""

    CSS = """
    /* qFlipper-like Theme with Flipper LED Highlights */

    Screen {
        layout: vertical; /* Changed to vertical to allow main-display and input to dock */
        overflow: hidden;
        background: #1A1A1A; /* Even darker background for contrast */
        color: #ff9722; /* qFlipper lightorange1 - Default text color */
    }

    #header { /* Add id for header */
        dock: top; /* Dock header to the top */
        width: 100%; /* Make header span full width */
    }

    #footer { /* Add id for footer */
       dock: bottom; /* Dock footer to the bottom */
       width: 100%; /* Make footer span full width */
    }

    Header {
        background: #1A1A1A; /* Even darker background - Same as background for consistency */
        /* Remove explicit white color to inherit default orange */
        text-style: bold;
    }

    Footer {
        background: #1A1A1A; /* Even darker background - Same as background for consistency */
        /* Remove explicit white color to inherit default orange */
    }

    #app-grid { /* Container for main display and sidebar */
        layout: grid;
        grid-size: 2;
        grid-columns: 2fr 1fr; /* Main display takes 2/3, sidebar takes 1/3 */
        height: 1fr; /* Allow app-grid to take remaining vertical space */
        width: 100%; /* Allow app-grid to take full horizontal space */
    }

    #main-display {
        height: 100%; /* Fill the height of the app-grid cell */
        width: 100%; /* Fill the width of the app-grid cell */
        border: none; /* Remove border for cleaner look */
    }
    RichLog {
        background: #1A1A1A; /* Even darker background */
        color: #ff9722; /* qFlipper lightorange1 - Default text color in log */
        min-height: 10; /* Ensure a minimum height */
        height: auto; /* Allow height to adjust based on content */
        padding: 0 1 0 1; /* Add some padding */
    }

    Input {
        dock: bottom; /* Keep input docked at the bottom */
        width: 100%;
        border: heavy #2ed832; /* qFlipper lightgreen - Full border around input */
        border-bottom: heavy #2ed832; /* Explicitly set bottom border */
        margin: 0 0 1 0; /* Add 1 unit margin to the bottom */
        height: 3; /* Give input some height */
        background: #1A1A1A; /* Even darker background */
        color: #2ed832; /* qFlipper lightgreen - User input text color */
    }

    /* Color classes for different message types */
    .fg-orange { color: #ff9722; } /* qFlipper lightorange1 */
    .fg-green { color: #2ed832; } /* qFlipper lightgreen */
    .fg-red { color: #ff5b27; } /* qFlipper lightred1 */
    .fg-blue { color: #228cff; } /* qFlipper lightblue */
    .bg-dark { background: #1A1A1A; /* Even darker background */ }

    /* Semantic classes (optional styling, primary color set by fg classes) */
    .command { text-style: bold; }
    .response { }
    .info { }
    .error { text-style: bold; }
    .question { text-style: bold; }

    #sidebar { /* Added CSS for sidebar */
        height: 100%; /* Make sidebar fill the height of its grid cell */
    }
    """

    # Update init to accept new agent loop, human interaction handler, flipper_agent, and llm_agent
    def __init__(
        self,
        agent_loop: AgentLoop,
        human_interaction_handler: HumanInteractionHandler,
        flipper_agent: Any, # Accept flipper_agent
        llm_agent: UnifiedLLMAgent # Accept llm_agent
    ):
        super().__init__()
        self.agent_loop = agent_loop
        self.human_interaction_handler = human_interaction_handler
        self.flipper_agent = flipper_agent # Assign flipper_agent
        self.llm_agent = llm_agent # Assign llm_agent
        self.agent_state = agent_loop.agent_state # Access state via the loop
        
        # Pass the app instance (self) to components that need it for UI updates
        self.agent_loop.app_instance = self
        self.human_interaction_handler.app_instance = self
        # Note: AgentState might also need app_instance if its methods display messages
        # self.agent_state.app_instance = self
        self.agent_loop.tool_executor.app_instance = self # Pass to tool executor

        self.input_widget = Input(placeholder="Enter command/query, type '/help' for assistance...")
        # Disable highlighting to prevent Textual from overriding colors for paths, numbers, etc.
        # Enable selectable text for copying
        self.main_display = RichLog(id="main-display", wrap=True, highlight=False, markup=True) # Use RichLog
        self.main_display.selectable = True # Enable text selection after creation
        self.main_display.can_focus = True # Allow the RichLog to receive focus
        # self.task_in_progress = var(False) # task_in_progress handled by AgentState/TaskManager now (line removed)

    def compose(self) -> ComposeResult:
        """Compose the application UI."""
        yield Header(id="header") # Add id to Header
        # Use a container to hold the main display and sidebar
        yield Container( # Remove id="app-grid" from this container, it will be applied via CSS
            self.main_display,
            TaskListTreeView(agent_state=self.agent_state, id="sidebar"),
            id="app-grid" # Add id to the container itself
        )
        yield self.input_widget
        yield Footer(id="footer") # Add id to Footer

    async def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Display the initial connection banner directly with ASCII art
        ascii_art = """
                                   __
                               _.-~  )
                    _..--~~~~,'   ,-/     _
                 .-'. . . .'   ,-','    ,' )
               ,'. . . _   ,--~,-'__..-'  ,'
             ,'. . .  (@)' ---~~~~      ,'
            /. . . . '~~             ,-'
           /. . . . .             ,-'
          ; . . . .  - .        ,'
         : . . . .       _     /
        . . . . .          `-.:
       . . . ./  - .          )
      .  . . |  _____..---.._/ 
~---~~~~----~~~~             ~~~---~~~~----~~~~
"""
        banner_content = f"{'─' * 79}\n" # Separator line above ASCII art
        banner_content += ascii_art + "\n"
        banner_content += f"{'─' * 79}\n" # Separator line below ASCII art
        # Define dynamic banner content with green color using hex markup
        port_str = f"[#2ed832]{self.flipper_agent.port}[/#2ed832]"
        provider_model_str = f"[#2ed832]{self.llm_agent.provider}/{self.llm_agent.model}[/#2ed832]"
        tokens_str = f"[#2ed832]{self.llm_agent.config.get('llm', {}).get('max_history_tokens', 'N/A')} tokens[/#2ed832]"

        # Construct the banner content
        banner_content = f"{'─' * 79}\n" # Separator line above ASCII art
        banner_content += ascii_art + "\n"
        banner_content += f"{'─' * 79}\n" # Separator line below ASCII art
        banner_content += f"✓ Flipper connect on port: {port_str}\n"
        banner_content += f"✓ LLM provider/model: {provider_model_str}\n"
        banner_content += f"✓ Context history: {tokens_str}\n"
        banner_content += f"{'─' * 79}\n"

        # Wrap the entire banner content in command class markup for consistent orange color
        await self.display_message(f"[command]{banner_content}[/command]")

    async def update_task_list_ui(self) -> None:
        """Updates the task list display in the sidebar."""
        task_list_widget = self.query_one("#sidebar", TaskListTreeView)
        task_list_widget.update_task_display()

    async def display_message(self, message: str) -> None:
        """Append a message with Textual markup to the main display and send to standard output."""
        # Write to the RichLog for display in the Textual UI
        self.main_display.write(message)
        
        # The message is now logged via the logger in main.py via display_message
        # No longer need to print directly to standard output

    # Add an asyncio.Event to signal when human input is received
    # This should ideally be initialized in __init__
    # self._human_input_received = asyncio.Event()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = message.value.strip()
        self.input_widget.value = ""  # Clear the input field

        if not user_input:
            return

        # Display user input (color coded for user)
        await self.display_message(f"[#2ed832]{'─' * 79}[/#2ed832]")
        await self.display_message(f"[#2ed832]> {user_input}[/#2ed832]")
        await self.display_message(f"[#2ed832]{'─' * 79}[/#2ed832]")


        logger.info(f"User input: {user_input}")

        if user_input.lower() in ('/exit', '/quit'):
            self.exit()  # Use exit() instead of shutdown() to exit the Textual app
            return

        if user_input.startswith('/'):
            # Handle special commands regardless of agent state
            await self.handle_special_commands(user_input)
            return

        # Check if the agent is currently awaiting human input
        if self.agent_state.awaiting_human_input:
            # Process the human input through the state
            # This will also add the human input to history
            human_input_result = self.agent_state.process_human_input(user_input)

            # Signal the AgentLoop worker that input is ready
            # Clear the event first in case it was already set
            self._human_input_received.clear()
            self._human_input_received.set()

            # Optional: Change input box appearance back if it was modified
            # self.input_widget.disabled = False # Example
            
            # The AgentLoop will resume processing on its next iteration or after awaiting the event
            await self.display_message(f"[yellow]Received human input. Resuming agent loop...[/]")
            return
        else:
            # This is a new user request
            await self.display_message(f"[yellow]Processing new request: '{user_input}'...[/]")

            # Start a worker to run the agent loop for this request
            # The worker will run the main agent loop function
            self.run_worker(
                self.agent_loop.process_user_request(user_input),
                name=f"agent_loop_worker_{time.time()}", # Unique name for the worker
                group="agent_processing"
            )

    # Need to add _human_input_received = asyncio.Event() to __init__

    async def handle_special_commands(self, input_text: str) -> None:
        """Process slash commands within the Textual app."""
        command = input_text[1:].lower()
        # Special commands might need to be handled differently if agent is awaiting human input
        # For now, allow them to interrupt or be handled in parallel?
        # Let's keep it simple: special commands are handled regardless of state.
        if command == 'help':
            await self.show_help()
        elif command == 'exit' or command == 'quit':
            self.exit()
        elif command == 'tasks': # Changed command name here
            await self.toggle_task_sidebar() # Call the toggle method
        # elif command == 'ask': # The 'ask' command is replaced by the agent's HITL flow
        #     await self.switch_to_ask_mode() # This old logic is removed
        else:
            await self.handle_unknown_command(command)
        # return is implicit

    async def toggle_task_sidebar(self) -> None:
        """Toggles the visibility of the task list sidebar and adjusts layout."""
        try:
            task_list_widget = self.query_one("#sidebar", TaskListTreeView)
            app_grid_container = self.query_one("#app-grid", Container) # Get the app-grid container

            if task_list_widget.styles.display == "none":
                # Sidebar is hidden, show it and revert grid
                task_list_widget.styles.display = ""
                app_grid_container.styles.grid_columns = "2fr 1fr" # Revert to two columns
                await self.display_message("[green]Task sidebar shown.[/green]")
            else:
                # Sidebar is visible, hide it and expand main display
                task_list_widget.styles.display = "none"
                app_grid_container.styles.grid_columns = "1fr" # Expand main display to full width
                await self.display_message("[green]Task sidebar hidden.[/green]")
        except Exception as e:
            await self.display_message(f"[red]Error toggling task sidebar: {e}[/red]")


    # Remove the old switch_to_ask_mode function
    # async def switch_to_ask_mode(self) -> None: ...

    async def show_help(self) -> None:
        """Display available commands."""
        help_text = f"\n[green]Available Commands:[/]\n"
        help_text += f"  [green]/exit, /quit[/] - Terminate the program\n"
        help_text += f"  [green]/tasks[/] - Show or hide the task list sidebar\n" # Changed command name in help
        # help_text += f"  [green]/ask[/] - Enter question/answer mode\n" # Remove old ask command help
        help_text += f"  [green]/help[/] - Show this help message"
        # Add help for interacting with the agent when awaiting human input?
        # e.g., how to respond to different request types. This might be handled in display_human_request.
        await self.display_message(help_text)
        # return is implicit
    async def show_help(self) -> None:
        """Display available commands."""
        help_text = f"\n[green]Available Commands:[/]\n"
        help_text += f"  [green]/exit, /quit[/] - Terminate the program\n"
        help_text += f"  [green]/ask[/] - Enter question/answer mode\n"
        help_text += f"  [green]/help[/] - Show this help message"
        await self.display_message(help_text)

    async def switch_to_ask_mode(self) -> None:
        """Handle mode transition."""
        await self.display_message(f"\nSwitching to inquiry mode...[/]")
        # TODO: Implement actual ask mode logic


    async def handle_unknown_command(self, command: str) -> None:
        """Handle invalid slash commands."""
        error_text = f"Unknown command: /{command}[/]\n"
        error_text += f"Type /help for supported commands"
        await self.display_message(error_text)


    async def display_human_request(self, request_type: str, prompt: str, options: Optional[List[str]] = None) -> None:
        """Display a request for human input."""
        # Format the prompt
        formatted_prompt = f"[bold #ff9722]👤 {request_type.capitalize()} Required:[/]\n{prompt}\n"
        
        # Format options if provided
        if options:
            formatted_prompt += "\nOptions:\n"
            for i, option in enumerate(options, 1):
                formatted_prompt += f"  {i}. {option}\n"
                
        await self.display_message(formatted_prompt)
        
        # For approval requests, add extra formatting
        if request_type == "approval":
            await self.display_message("[#2ed832]Please type 'approve', 'reject', or 'modify'[/#2ed832]")
        elif options:
            await self.display_message("[#2ed832]Please type your choice (number or text)[/#2ed832]")


def run_interactive_loop(
    agent_loop: AgentLoop,
    human_interaction_handler: HumanInteractionHandler,
    flipper_agent: Any, # Accept flipper_agent
    llm_agent: UnifiedLLMAgent # Accept llm_agent
):
    """Run the main user interaction loop using Textual."""
    # Pass the new agent_loop, human_interaction_handler, flipper_agent, and llm_agent instances to the app
    app = AgentFlipper(agent_loop, human_interaction_handler, flipper_agent, llm_agent)
    app.run()