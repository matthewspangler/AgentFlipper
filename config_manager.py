"""
Manages application configuration loading and parsing.
"""

import argparse
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from colors import Colors

logger = logging.getLogger("FlipperAgent")

# Default configuration
CONFIG_FILE = os.path.expanduser("~/.config/flipper_agent/config.yaml")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
# Note: VECTOR_STORE_PATH is used by RAGRetriever, so it should probably stay there
# VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "flipper_cli_faiss")


def load_config() -> Dict[str, Any]:
    """Load configuration from config file"""
    # First check for config in the home directory
    config_path = Path(CONFIG_FILE)

    # If not found in home directory, check for config in the current directory
    if not config_path.exists():
        config_path = Path(DEFAULT_CONFIG_PATH)

    if not config_path.exists():
        # Create default config if it doesn't exist
        default_config = {
            "flipper": {
                "port": "/dev/ttyACM0",  # Default port for Flipper Zero
                "timeout": 5.0
            },
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5-coder:14b",
                "api_base": "http://localhost:11434",
                "max_history_tokens": 6000,
                "max_recursion_depth": 10,  # Maximum recursion depth for command loops
                "user_prompt": "You are an assistant controlling a Flipper Zero device."
                               "Provide helpful and accurate commands based on the user's request."
            }
        }

        # Create parent directories if they don't exist
        Path(DEFAULT_CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)

        with open(DEFAULT_CONFIG_PATH, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        print(f"{Colors.GREEN}✓ Loaded config from {config_path}{Colors.ENDC}")
        return config

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments"""
    parser = argparse.ArgumentParser(description='Flipper Zero AI Agent using PyFlipper and RAG')
    parser.add_argument('--port', type=str, help='Serial port for Flipper Zero')
    parser.add_argument('--config', type=str, help='Path to config file')
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

def load_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configuration from files and command-line"""
    config = load_config_from_file(args.config) if args.config else load_config()
    apply_command_line_overrides(config, args)
    return config

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from specified file path"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config

def apply_command_line_overrides(config: Dict[str, Any], args: argparse.Namespace):
    """Apply command-line parameter overrides to configuration"""
    if args.port:
        config["flipper"]["port"] = args.port
        logger.info(f"Command-line override for port: {args.port}")

    if args.model:
        handle_model_override(config, args.model)

def handle_model_override(config: Dict[str, Any], model_spec: str):
    """Process model specification from command line"""
    if '/' in model_spec:
        provider, model = model_spec.split('/', 1)
        config["llm"]["provider"] = provider
        config["llm"]["model"] = model
    else:
        config["llm"]["model"] = model_spec
    logger.info(f"Command-line override for model: {model_spec}")