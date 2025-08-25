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
from agent.llm_agent import UnifiedLLMAgent
from agent.agent_state import AgentState

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

def configure_llm_agent(config: Dict[str, Any], agent_state: AgentState,
                        args: argparse.Namespace) -> UnifiedLLMAgent:
    """Initialize and configure the LLM agent with runtime parameters."""
    agent = UnifiedLLMAgent(config, agent_state)

    return agent