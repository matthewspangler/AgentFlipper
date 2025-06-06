"""
Handles user interface interactions.
"""

import sys
import logging
from typing import Dict, Any

from colors import Colors
from hardware_manager import FlipperZeroManager # Assuming FlipperZeroManager is in hardware_manager.py
from llm_agent import LLMAgent # Assuming LLMAgent is in llm_agent.py
from request_processor import process_user_request # Assuming process_user_request is in request_processor.py


logger = logging.getLogger("FlipperAgent")

def display_connection_banner(config: Dict[str, Any], agent: LLMAgent):
    """Show startup connection status and configuration"""
    print(f"{Colors.GREEN}{'─' * 58}")
    print(f"✓ Connected to Flipper Zero on {config['flipper']['port']:^35}")
    print(f"✓ LLM Provider/Model: {agent.provider}/{agent.model_name:^35}")
    print(f"✓ Context History: {agent.max_history_tokens} tokens{'':^20}")
    # LOG_FILE needs to be imported or passed in
    # print(f"✓ Log File: {LOG_FILE:^45}") 
    print(f"{'─' * 58}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Ready for commands (type '/help' for assistance){Colors.ENDC}")

def run_interactive_loop(flipper_agent: FlipperZeroManager, llm_agent: LLMAgent):
    """Manage the main user interaction loop"""
    try:
        while True:
            handle_user_input(flipper_agent, llm_agent)
    except KeyboardInterrupt:
        logger.info("Graceful shutdown initiated")
        print(f"\n{Colors.ORANGE}Exiting gracefully...{Colors.ENDC}")
    except Exception as e:
        logger.exception(f"Critical runtime error: {str(e)}")
        print(f"{Colors.FAIL}Fatal error: {str(e)}{Colors.ENDC}")
    finally:
        flipper_agent.disconnect()
        logger.info("Session terminated cleanly")

def handle_user_input(flipper_agent: FlipperZeroManager, llm_agent: LLMAgent):
    """Process a single user input iteration"""
    if llm_agent.task_in_progress:
        print(f"\n{Colors.BOLD}Continuing: {llm_agent.task_description}{Colors.ENDC}")

    user_input = get_user_input()

    if is_exit_command(user_input):
        sys.exit(0)

    if handle_special_commands(user_input):
        return

    process_user_request(user_input, flipper_agent, llm_agent, 0)

def get_user_input() -> str:
    """Display input prompt and retrieve user input"""
    print(f"\n{Colors.BLUE}{'═' * 60}{Colors.ENDC}")
    try:
        return input(f"{Colors.BLUE}> {Colors.ENDC}").strip()
    finally:
        print(f"{Colors.BLUE}{'═' * 60}{Colors.ENDC}")

def is_exit_command(input_text: str) -> bool:
    """Check for termination commands"""
    return input_text.lower() in ('exit', 'quit')

def handle_special_commands(input_text: str) -> bool:
    """Process slash commands and return True if handled"""
    if not input_text.startswith('/'):
        return False

    command = input_text[1:].lower()
    command_handlers = {
        'help': show_help,
        'ask': switch_to_ask_mode,
        'exit': lambda: sys.exit(0),
        'quit': lambda: sys.exit(0)
    }

    handler = command_handlers.get(command, handle_unknown_command)
    handler()
    return True

def show_help():
    """Display available commands"""
    print(f"\n{Colors.GREEN}Available Commands:{Colors.ENDC}")
    print(f"  {Colors.GREEN}/exit, /quit{Colors.ENDC} - Terminate the program")
    print(f"  {Colors.GREEN}/ask{Colors.ENDC} - Enter question/answer mode")
    print(f"  {Colors.GREEN}/help{Colors.ENDC} - Show this help message")

def switch_to_ask_mode():
    """Handle mode transition"""
    print(f"\n{Colors.PURPLE}Switching to inquiry mode...{Colors.ENDC}")

def handle_unknown_command(command: str):
    """Handle invalid slash commands"""
    print(f"{Colors.WARNING}Unknown command: /{command}{Colors.ENDC}")
    print(f"Type {Colors.BOLD}/help{Colors.ENDC} for supported commands")