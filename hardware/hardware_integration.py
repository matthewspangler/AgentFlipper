"""
Handles hardware integration and initialization.
"""

import argparse
import logging
import sys # Added sys import
from typing import Dict, Any, Optional

from ui import Colors
from hardware.hardware_manager import FlipperZeroManager
from rag.rag_retriever import RAGRetriever
from agent.llm_agent import UnifiedLLMAgent # Import UnifiedLLMAgent
from agent.agent_state import AgentState # Import AgentState
# Assuming AgentState needs to be imported here if not available via other imports
# from .agent_loop.agent_state import AgentState # Uncomment if needed

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

# Assuming AgentState is passed in from the calling code (e.g., main.py)
# If AgentState is managed and accessible globally or via another service locator pattern,
# this signature might not need to change, and we'd get agent_state differently.
# Based on UnifiedLLMAgent.__init__ requiring agent_state, passing it seems most direct.

def configure_llm_agent(config: Dict[str, Any], agent_state: AgentState, # Added agent_state parameter
                       args: argparse.Namespace) -> UnifiedLLMAgent: # Changed return type
    """Initialize and configure the LLM agent with runtime parameters."""
    # Instantiate UnifiedLLMAgent instead of LLMAgent
    # UnifiedLLMAgent constructor takes config and agent_state
    agent = UnifiedLLMAgent(config, agent_state)

    # The config dictionary passed likely already has overrides applied,
    # so UnifiedLLMAgent can access them directly.

    return agent