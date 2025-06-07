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

logger = logging.getLogger("FlipperAgent")

# Define process_request_in_worker as a standalone async function
# It now accepts the app_instance directly for UI updates
async def process_request_in_worker(
    app_instance: "TextualApp", # Add app_instance as an argument
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


class TextualApp(App):
    """Textual application for the Flipper Zero AI Agent."""

    CSS = """
    Screen {
        layout: vertical;
        overflow: hidden;
        background: #000000; /* Black */
        color: #FFFFFF; /* White - Default text color */
    }
    
    Header {
        background: #FF8A00; /* Orange */
        color: #FFFFFF; /* White */
        text-style: bold;
    }

    Footer {
        background: #FF8A00; /* Orange */
        color: #FFFFFF; /* White */
    }

    #main-display {
        height: 1fr;
        width: 100%;
        border: none; /* Remove border for cleaner look */
    }
    
    RichLog {
        background: #000000; /* Black */
        color: #FFFFFF; /* White - Default text color in log */
        min-height: 10; /* Ensure a minimum height */
        height: auto; /* Allow height to adjust based on content */
        padding: 0 1 0 1; /* Add some padding */
    }
    
    Input {
        dock: bottom;
        width: 100%;
        border-top: heavy #00FF00; /* Green border for input area */
        margin: 0;
        height: 3; /* Give input some height */
        background: #000000; /* Black */
        color: #FFFFFF; /* White */
    }

    Input:focus {
        border-top: heavy #FF8A00; /* Orange border when focused */
    }
    
    /* Color classes for different message types */
    .command {
        color: #FFFF00; /* Yellow - User input command */
        text-style: bold;
    }
    .response {
        color: #FFFFFF; /* White - Flipper response */
    }
    .info {
        color: #00FF00; /* Green - Info messages */
    }
    .error {
        color: #FF8A00; /* Orange - Error messages */
        text-style: bold;
    }
    .question {
        color: #8A2BE2; /* Purple - Questions */
        text-style: bold;
    }
    """

    def __init__(self, flipper_agent: FlipperZeroManager, llm_agent: LLMAgent):
        super().__init__()
        self.flipper_agent = flipper_agent
        self.llm_agent = llm_agent
        self.input_widget = Input(placeholder="Enter command or query...")
        self.main_display = RichLog(id="main-display", wrap=True, highlight=True, markup=True) # Use RichLog
        self.task_in_progress = var(False)

    def compose(self) -> ComposeResult:
        """Compose the application UI."""
        yield Header()
        yield self.main_display
        yield self.input_widget
        yield Footer()

    async def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Display the initial connection banner directly
        banner_content = f"[green]{'─' * 58}[/]\n"
        banner_content += f"[green]✓ Connected to Flipper Zero on {self.flipper_agent.port:^35}[/]\n"
        banner_content += f"[green]✓ LLM Provider/Model: {self.llm_agent.provider}/{self.llm_agent.model_name:^35}[/]\n"
        banner_content += f"[green]✓ Context History: {self.llm_agent.max_history_tokens} tokens{'':^20}[/]\n"
        banner_content += f"[green]{'─' * 58}[/]\n"
        banner_content += f"[green]✓ Ready for commands (type '/help' for assistance)[/]"
        await self.display_message(banner_content)

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

        await self.display_message(f"[blue bold]> {user_input}[/]")
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
        await self.display_message(f"\n[purple]Switching to inquiry mode...[/]")
        # TODO: Implement actual ask mode logic

    async def handle_unknown_command(self, command: str) -> None:
        """Handle invalid slash commands."""
        error_text = f"[yellow]Unknown command: /{command}[/]\n"
        error_text += f"Type [red bold]/help[/] for supported commands"
        await self.display_message(error_text)


def run_interactive_loop(flipper_agent: FlipperZeroManager, llm_agent: LLMAgent):
    """Run the main user interaction loop using Textual."""
    app = TextualApp(flipper_agent, llm_agent)
    app.run()