#!/usr/bin/env python3
"""
AgentFlipper using PyFlipper and RAG

This script connects the Flipper Zero to LiteLLM (configured to use local Ollama)
using the PyFlipper library as a backend, allowing users to interact with the
Flipper Zero using natural language. It uses RAG to improve command quality.
"""

import argparse
import os
import sys
import yaml
import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from rag import RAGRetriever 
from agent.request_processor import process_user_request
from hardware import establish_flipper_connection, initialize_rag_system, configure_llm_agent
from ui import Colors, AgentFlipper, run_interactive_loop
from agent import AgentLoop, AgentState, TaskManager, ToolExecutor
from hardware import HardwareManager, FlipperZeroManager 
from agent.llm_agent import UnifiedLLMAgent
from ui import HumanInteractionHandler, Colors

# Custom exception for configuration errors.
class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"agent_flipper_{time.strftime('%Y%m%d_%H%M%S')}.log")
from hardware.hardware_manager import HardwareManager # Import the new HardwareManager class
AI_LOG_FILE = os.path.join(LOG_DIR, f"agent_flipper_ai_{time.strftime('%Y%m%d_%H%M%S')}.log")

# Configure main logger - default to file only
logger = logging.getLogger("AgentFlipper")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add file handler for main logger
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configure AI response logger
ai_logger = logging.getLogger("AgentFlipperAI")
ai_logger.setLevel(logging.DEBUG)
ai_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add file handler for AI logger
ai_file_handler = logging.FileHandler(AI_LOG_FILE)
ai_file_handler.setFormatter(ai_formatter)
ai_logger.addHandler(ai_file_handler)
# Functions for configuration loading and parsing (adapted from config_manager.py)

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from specified file path, handling empty/invalid files."""
    logger.debug(f"Attempting to load config from specified file: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            logger.warning(f"Config file {config_path} is empty or contains invalid YAML. Returning empty dict.")
            return {}
        if not isinstance(config_data, dict):
            logger.warning(f"Config file {config_path} content is not a dictionary. Returning empty dict.")
            return {}
        logger.info(f"Loaded config from {config_path}")
        return config_data

    except FileNotFoundError:
        logger.debug(f"Config file not found at {config_path}. Returning empty dict.")
        return {}
    except yaml.YAMLError as e:
        # Raise ConfigError for YAML parsing issues
        raise ConfigError(f"Error parsing YAML config file {config_path}: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during file loading
        raise ConfigError(f"An unexpected error occurred while loading config file {config_path}: {e}") from e


def load_from_default_locations() -> Dict[str, Any]:
    """Load configuration from default locations: user home dir or project dir."""
    user_config_path = Path(os.path.expanduser("~/.config/agent_flipper/config.yaml"))
    project_config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))

    config_data = {}
    # Load from user config first if it exists
    if user_config_path.exists():
        logger.debug(f"Checking user config at: {user_config_path}")
        user_data = load_config_from_file(str(user_config_path))
        # Merge user data into config_data
        if isinstance(user_data, dict):
            config_data.update(user_data)

    # Then load from project config if it exists and user config didn't provide data
    if project_config_path.exists():
         logger.debug(f"Checking project config at: {project_config_path}")
         project_data = load_config_from_file(str(project_config_path))
         # Merge project data into config_data (user data takes precedence)
         if isinstance(project_data, dict):
             # A simple update might overwrite. For merging nested, need a helper.
             # Let's implement a simple recursive merge.
             def recursive_merge(target, source):
                 for key, value in source.items():
                     if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                         recursive_merge(target[key], value)
                     else:
                         target[key] = value
             recursive_merge(config_data, project_data)


    if not config_data:
         logger.warning("No valid config file found in default locations. Starting with empty configuration.")

    return config_data

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    logger.debug("Parsing command-line arguments.")
    parser = argparse.ArgumentParser(description='AgentFlipper using PyFlipper and RAG')
    parser.add_argument('--port', type=str, help='Serial port for Flipper Zero')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--model', type=str, help='LLM model to use (format: provider/model or just model name)')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG system')
    parser.add_argument('--max-history-tokens', type=int,
                       help='Maximum token count for conversation history')
    parser.add_argument('--max-recursion-depth', type=int,
                       help='Maximum recursion depth for command loops')
    parser.add_argument('--log-level', type=str,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set logging level')
    return parser.parse_args()


def load_and_merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from files and command-line arguments."""
    logger.debug("Loading and merging configuration.")
    try:
        # Load base config data from file (either custom or default locations)
        if args.config:
            base_config_data = load_config_from_file(args.config)
        else:
            base_config_data = load_from_default_locations()

        # Create a mutable copy to apply overrides
        merged_config = base_config_data.copy()

        # Apply command-line overrides
        override_data = {}
        if args.port:
            override_data.setdefault("flipper", {})["port"] = args.port
            logger.debug(f"Command-line override for flipper.port: {args.port}")

        # args.config is handled above, no need to process here
        # if args.config: pass

        if args.model:
            if '/' in args.model:
                provider, model = args.model.split('/', 1)
                override_data.setdefault("llm", {})["provider"] = provider
                override_data.setdefault("llm", {})["model"] = model
                logger.debug(f"Command-line override for llm.provider: {provider}, llm.model: {model}")
            else:
                override_data.setdefault("llm", {})["model"] = args.model
                logger.debug(f"Command-line override for llm.model: {args.model}")

        # args.no_rag is a flag, likely handled where RAG is initialized
        # if args.no_rag is not None: pass

        if args.max_history_tokens is not None:
            override_data.setdefault("llm", {})["max_history_tokens"] = args.max_history_tokens
            logger.debug(f"Command-line override for llm.max_history_tokens: {args.max_history_tokens}")

        if args.max_recursion_depth is not None:
             override_data.setdefault("llm", {})["max_recursion_depth"] = args.max_recursion_depth
             logger.debug(f"Command-line override for llm.max_recursion_depth: {args.max_recursion_depth}")

        # args.log_level is for logging setup, likely handled separately
        # if args.log_level is not None: pass

        # Merge override data into the base config
        def recursive_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    recursive_merge(target[key], value)
                else:
                    target[key] = value
        recursive_merge(merged_config, override_data)

        logger.debug("Configuration loaded and merged successfully.")
        return merged_config

    except ConfigError as e:
        logger.critical(f"Configuration Error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during the loading process
        logger.critical(f"An unexpected error occurred during configuration loading: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Orchestrate the application startup and shutdown"""
    args = parse_arguments()
    config = load_and_merge_config(args)

    # Initialize AgentState first
    agent_state = AgentState(config=config)

    # Initialize components that depend on AgentState but not immediately on the UI
    task_manager = TaskManager(agent_state=agent_state)

    # Initialize the Flipper Zero manager for hardware interaction
    # Use FlipperZeroManager instead of the base HardwareManager
    flipper_agent = FlipperZeroManager(port=config.get("flipper", {}).get("port")) # Instantiate FlipperZeroManager
    
    # Establish connection to the device before proceeding
    if not flipper_agent.connect():
        logger.error(f"Failed to connect to Flipper Zero on port {config.get('flipper', {}).get('port')}")
        print(f"{Colors.FAIL}Connection failed - check device and permissions{Colors.ENDC}")
        sys.exit(1)
    
    logger.info("Successfully connected to Flipper Zero")
    agent_state.flipper_agent = flipper_agent # Pass flipper_agent to state

    # The new UnifiedLLMAgent will handle LLM calls and potentially use RAG internally
    llm_agent = UnifiedLLMAgent(config=config, agent_state=agent_state)
    agent_state.llm_agent = llm_agent # Pass llm_agent to state

    # Initialize components that depend on AgentState and will depend on the UI (app_instance)
    # Pass None for app_instance initially, it will be set by the UI
    human_interaction_handler = HumanInteractionHandler(agent_state=agent_state, app_instance=None)
    tool_executor = ToolExecutor(agent_state=agent_state, app_instance=None)

    # Initialize the main agent loop, passing the core components
    agent_loop = AgentLoop(
        agent_state=agent_state,
        task_manager=task_manager,
        tool_executor=tool_executor,
        llm_agent=llm_agent,
        human_interaction_handler=human_interaction_handler,
        app_instance=None # App instance set by the UI after creation
    )

    # Instantiate the Textual App (AgentFlipper) and pass the loop and handler
    # The App's __init__ should handle setting the app_instance back to components
    app = AgentFlipper(
        agent_loop=agent_loop,
        human_interaction_handler=human_interaction_handler,
        flipper_agent=flipper_agent, # Pass flipper_agent
        llm_agent=llm_agent # Pass llm_agent
    )

    # Run the interactive loop which starts the Textual app
    run_interactive_loop(
        agent_loop,
        human_interaction_handler,
        flipper_agent, # Pass flipper_agent
        llm_agent # Pass llm_agent
    )


if __name__ == "__main__":
    # Add a simple connection test before running the main app
    try:
        logger.info("Running main application")
        main()
    except Exception as e:
        logger.critical(f"Fatal error in main application: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)