"""
Class for retrieving relevant documentation using RAG
"""

import os
import logging
import argparse
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from pathlib import Path

from ui.colors import Colors # Import the Colors class

logger = logging.getLogger("FlipperAgent")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "flipper_cli_faiss")

class RAGRetriever:
    """Class for retrieving relevant documentation using RAG"""

    def __init__(self):
        """Initialize the RAG retriever"""
        self.vector_store = None
        self.embeddings = None
        self.initialized = False

    def initialize(self, args: argparse.Namespace) -> bool:
        """Initialize the vector store and embeddings"""
        try:
            # Check if vector store exists
            if not os.path.exists(VECTOR_STORE_PATH):
                logger.warning(f"Vector store not found at {VECTOR_STORE_PATH}")
                print(f"Vector store not found at {VECTOR_STORE_PATH}")
                print("Please run flipper_docs_loader.py first to create the vector store")
                return False

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            # Load the vector store
            self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            self.initialized = True
            logger.info("RAG system initialized successfully")
            print(f"{Colors.GREEN}✓ RAG system initialized successfully{Colors.ENDC}")
            return True
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            print(f"Error initializing RAG system: {str(e)}")
            return False

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documentation based on the query"""
        if not self.initialized:
            logger.warning("RAG system not initialized")
            print("RAG system not initialized")
            return []

        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            print(f"Error retrieving documents: {str(e)}")
            return []