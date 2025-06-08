"""
Handles hardware integration and initialization.
"""

import argparse
import logging
import sys # Added sys import
from typing import Dict, Any, Optional

from colors import Colors
from hardware_manager import FlipperZeroManager
from rag_retriever import RAGRetriever
from llm_agent import LLMAgent

logger = logging.getLogger("AgentFlipper")

def establish_flipper_connection(config: Dict[str, Any]) -> FlipperZeroManager:
    """Initialize and validate Flipper Zero connection."""
    # Access config directly, with checks
    flipper_config = config.get("flipper", {})
    port = flipper_config.get("port") # Use .get for safety

    if not port:
        logger.error("Flipper port is not specified in the configuration.")
        print(f"{Colors.FAIL}Error: Flipper port is not specified in the configuration.{Colors.ENDC}")
        sys.exit(1)

    agent = FlipperZeroManager(port=port)
    if not agent.connect():
        logger.error(f"Flipper Zero connection failed on port {port}")
        print(f"{Colors.FAIL}Connection failed on port {port} - check device and permissions{Colors.ENDC}")
        sys.exit(1)
    return agent

def initialize_rag_system(args: argparse.Namespace) -> Optional[RAGRetriever]:
    """Configure the RAG retrieval system if enabled"""
    if args.no_rag:
        logger.info("RAG system disabled via command line")
        return None

    print(f"{Colors.PURPLE}Initializing RAG knowledge base...{Colors.ENDC}")
    rag = RAGRetriever()
    return rag if rag.initialize(args) else None

def configure_llm_agent(config: Dict[str, Any], rag: Optional[RAGRetriever],
                       args: argparse.Namespace) -> LLMAgent:
    """Initialize and configure the LLM agent with runtime parameters."""
    # Pass the raw config dictionary to LLMAgent
    agent = LLMAgent(config, rag)

    # These overrides were previously applied by ConfigManager.
    # LLMAgent will now access them directly from the passed config dict.
    # No changes needed here, as the config dict passed already has overrides applied.

    return agent