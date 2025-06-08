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
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import Container
from textual.reactive import var
from textual.worker import Worker, get_current_worker

from hardware_manager import FlipperZeroManager
from llm_agent import LLMAgent
from request_processor import process_user_request

logger = logging.getLogger("AgentFlipper")

# Define process_request_in_worker as a standalone async function
# It now accepts the app_instance directly for UI updates
async def process_request_in_worker(
    app_instance: "AgentFlipper", # Add app_instance as an argument
    user_input: str,
    flipper_agent: FlipperZeroManager,
    llm_agent: LLMAgent,
    recursion_depth: int = 0
) -> None:
    """Worker function to process user requests."""
    logger.info("Worker started for processing request.")
    try:
        # Process the user request - AWAITING the async function
        # Pass the app_instance directly for UI updates
        await process_user_request(
            app_instance, # Pass app_instance
            user_input,
            flipper_agent,
            llm_agent,
            recursion_depth
        )
        logger.info(f"Worker completed processing request: {user_input}")
    except Exception as e:
        error_message = f"Error in worker: {str(e)}"
        logger.error(error_message, exc_info=True)
        # More robust sanitization for error messages: escape [ and ] with backslashes
        sanitized_error_message = error_message.replace('[', '\\[').replace(']', '\\]')
        await app_instance.display_message(f"[red]{sanitized_error_message}[/]") # Use app_instance.display_message


class AgentFlipper(App):
    """Textual application for the AgentFlipper."""

    CSS = """
    /* qFlipper-like Theme with Flipper LED Highlights */

    Screen {
        layout: vertical;
        overflow: hidden;
        background: #1A1A1A; /* Even darker background for contrast */
        color: #ff9722; /* qFlipper lightorange1 - Default text color */
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

    #main-display {
        height: 1fr;
        width: 100%;
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
        dock: bottom;
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
    .bg-dark { background: #1A1A1A; } /* Even darker background */

    /* Semantic classes (optional styling, primary color set by fg classes) */
    .command { text-style: bold; }
    .response { }
    .info { }
    .error { text-style: bold; }
    .question { text-style: bold; }
    """

    def __init__(self, flipper_agent: FlipperZeroManager, llm_agent: LLMAgent):
        super().__init__()
        self.flipper_agent = flipper_agent
        self.llm_agent = llm_agent
        self.input_widget = Input(placeholder="Enter command/query, type '/help' for assistance...")
        # Disable highlighting to prevent Textual from overriding colors for paths, numbers, etc.
        self.main_display = RichLog(id="main-display", wrap=True, highlight=False, markup=True) # Use RichLog
        self.task_in_progress = var(False)

    def compose(self) -> ComposeResult:
        """Compose the application UI."""
        yield Header()
        yield self.main_display
        yield self.input_widget
        yield Footer()

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
        provider_model_str = f"[#2ed832]{self.llm_agent.provider}/{self.llm_agent.model_name}[/#2ed832]"
        tokens_str = f"[#2ed832]{self.llm_agent.max_history_tokens} tokens[/#2ed832]"

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

    async def display_message(self, message: str) -> None:
        """Append a message with Textual markup to the main display."""
        # RichLog handles rich/textual markup and scrolling automatically
        self.main_display.write(message)

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = message.value.strip()
        self.input_widget.value = ""  # Clear the input field

        if not user_input:
            return

        # Add green separator above user input display
        await self.display_message(f"[#2ed832]{'─' * 79}[/#2ed832]")
        await self.display_message(f"[#2ed832]> {user_input}[/#2ed832]")
        # Add green separator below user input display
        await self.display_message(f"[#2ed832]{'─' * 79}[/#2ed832]")

        logger.info(f"User input: {user_input}")

        if user_input.lower() in ('/exit', '/quit'):
            self.exit()  # Use exit() instead of shutdown() to exit the Textual app
            return

        if user_input.startswith('/'):
            await self.handle_special_commands(user_input)
            return

        # Process the request
        await self.display_message(f"[yellow]Processing request: '{user_input}'...[/]")

        # Create a partial function with arguments pre-bound
        worker_id = f"request_worker_{time.time()}"
        # Pass the app instance directly to the worker function
        bound_worker = partial(
            process_request_in_worker,
            self, # Pass the app instance
            user_input,
            self.flipper_agent,
            self.llm_agent,
            0
        )

        # Pass only the bound worker function and Textual-specific kwargs
        self.run_worker(
            bound_worker,
            name=worker_id,
            group="request_processing"
        )

    async def handle_special_commands(self, input_text: str) -> None:
        """Process slash commands within the Textual app."""
        command = input_text[1:].lower()
        if command == 'help':
            await self.show_help()
        elif command == 'ask':
            await self.switch_to_ask_mode()
        else:
            await self.handle_unknown_command(command)

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


def run_interactive_loop(flipper_agent: FlipperZeroManager, llm_agent: LLMAgent):
    """Run the main user interaction loop using Textual."""
    app = AgentFlipper(flipper_agent, llm_agent)
    app.run()