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

logger = logging.getLogger("FlipperAgent")

def establish_flipper_connection(config: Dict[str, Any]) -> FlipperZeroManager:
    """Initialize and validate Flipper Zero connection"""
    agent = FlipperZeroManager(port=config["flipper"]["port"])
    if not agent.connect():
        logger.error("Flipper Zero connection failed")
        print(f"{Colors.FAIL}Connection failed - check device and permissions{Colors.ENDC}")
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
    """Initialize and configure the LLM agent with runtime parameters"""
    agent = LLMAgent(config["llm"], rag)

    if args.max_history_tokens:
        agent.max_history_tokens = args.max_history_tokens
        logger.info(f"Set history token limit: {args.max_history_tokens}")

    if args.max_recursion_depth:
        agent.max_recursion_depth = args.max_recursion_depth
        logger.info(f"Set maximum recursion depth: {args.max_recursion_depth}")

    return agent