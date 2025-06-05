#!/usr/bin/env python3
"""
Flipper Zero AI Agent using PyFlipper and RAG

This script connects the Flipper Zero to LiteLLM (configured to use local Ollama)
using the PyFlipper library as a backend, allowing users to interact with the
Flipper Zero using natural language. It uses RAG to improve command quality.
"""

import argparse
import os
import sys
import yaml
import re
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"flipper_agent_{time.strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging - default to file only
logger = logging.getLogger("FlipperAgent")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add file handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler will be added based on command line option

# Path setup for PyFlipper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "pyFlipper/src"))

# Import PyFlipper
try:
    from pyflipper.pyflipper import PyFlipper
except ImportError:
    logger.error("PyFlipper module not found. Make sure the pyFlipper submodule is initialized.")
    print("Error: PyFlipper module not found. Make sure the pyFlipper submodule is initialized.")
    sys.exit(1)

# Import LiteLLM
try:
    from litellm import completion
except ImportError:
    logger.error("litellm package not found. Please install it with 'pip install litellm'")
    print("Error: litellm package not found. Please install it with 'pip install litellm'")
    sys.exit(1)

# Import LangChain
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.schema import Document
except ImportError:
    logger.error("langchain packages not found. Please install with 'pip install langchain langchain-community sentence-transformers'")
    print("Error: langchain packages not found. Please install with 'pip install langchain langchain-community sentence-transformers'")
    sys.exit(1)

# Default configuration
CONFIG_FILE = os.path.expanduser("~/.config/flipper_agent/config.yaml")
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
VECTOR_STORE_PATH = os.path.join(SCRIPT_DIR, "docs", "flipper_cli_faiss")

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'   # For loading/progress/waiting messages
    ORANGE = '\033[38;5;208m'
    WARNING = '\033[95m'  # Use purple for warnings
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

class HardwareManager:
    """Base class for hardware device management"""
    
    def __init__(self, port: str):
        self.port = port
        self.connection = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Return connection status"""
        return self._connected

    def connect(self) -> bool:
        """Establish connection to hardware device"""
        raise NotImplementedError

    def disconnect(self):
        """Close hardware connection and cleanup resources"""
        if self.connection:
            try:
                self.connection.close()
                logger.debug("Hardware connection closed gracefully")
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
        self.connection = None
        self._connected = False
        logger.info("Hardware connection terminated")

class FlipperZeroManager(HardwareManager):
    """Manages Flipper Zero device communication using PyFlipper"""
    
    def connect(self) -> bool:
        """Establish connection to Flipper Zero"""
        try:
            logger.info(f"Connecting to Flipper Zero on {self.port}")
            self.connection = PyFlipper(com=self.port)
            self.connection.device_info.info()  # Test connection
            self._connected = True
            logger.info("Flipper Zero connection established")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.disconnect()
            return False

    def send_command(self, command: str) -> str:
        """Execute a command on the Flipper Zero device"""
        if not self.is_connected:
            raise ConnectionError("Device not connected")
            
        try:
            logger.info(f"Executing command: {command}")
            return self.connection._serial_wrapper.send(command)
        except Exception as e:
            logger.error(f"Command failed: {str(e)}")
            self.disconnect()
            raise
        if not self.flipper:
            logger.error("Not connected to Flipper Zero")
            return "Error: Not connected to Flipper Zero"
        
        try:
            # Display command in orange color with separator
            print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ Sent command -> {command}{Colors.ENDC}")
            print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
            logger.info(f"Executing command: {command}")
            
            # Access the private serial wrapper to send arbitrary commands
            response = self.flipper._serial_wrapper.send(command)
            logger.info(f"Command response: {response}")
            return response if response.strip() else "no device response"
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def execute_commands(self, commands: List[str]) -> List[Tuple[str, str]]:
        """
        Execute multiple commands in sequence and return a list of (command, response) tuples.
        This allows tracking what commands were executed and their responses.
        """
        results = []
        if not commands:
            logger.warning("No commands to execute")
            return results
            
        # Log all commands before execution for debugging
        logger.info(f"Commands to execute: {commands}")
            
        for cmd in commands:
            # Skip empty commands or special markers
            cmd = cmd.strip()
            if not cmd or cmd.startswith("INFO:") or cmd.startswith(":"):
                logger.warning(f"Skipping invalid command: {cmd}")
                continue
            
            # Execute command and get response
            response = self.send_command(cmd)
            results.append((cmd, response))
            
            # Print the response with clearer formatting
            print(f"{Colors.ORANGE}{'─' * 50}{Colors.ENDC}")
            print(f"{Colors.ORANGE}# Device Response:{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.ORANGE}{response}{Colors.ENDC}")
            
            # Add single separator between agent actions
            print(f"{Colors.ORANGE}{'─' * 50}{Colors.ENDC}")
        
        logger.info(f"Executed {len(results)} commands")
        return results
    
    def disconnect(self):
        """Disconnect from the Flipper Zero"""
        # PyFlipper doesn't have an explicit disconnect method,
        # but the serial port will be closed when the object is deleted
        self.flipper = None
        logger.info("Disconnected from Flipper Zero")

class RAGRetriever:
    """Class for retrieving relevant documentation using RAG"""
    
    def __init__(self):
        """Initialize the RAG retriever"""
        self.vector_store = None
        self.embeddings = None
        self.initialized = False
    
    def initialize(self) -> bool:
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

def parse_commands(command_text: str) -> List[str]:
    """
    Parse a command string into individual commands.
    Commands can be separated by newlines or semicolons.
    """
    # Handle empty input
    if not command_text or not command_text.strip():
        logger.warning("Empty command text received")
        return []
        
    # First, replace semicolons with newlines (unless they're in quotes)
    # This is a simplified approach - for complex parsing consider a proper parser
    in_quotes = False
    chars = []
    
    for char in command_text:
        if char == '"' or char == "'":
            in_quotes = not in_quotes
        if char == ';' and not in_quotes:
            chars.append('\n')
        else:
            chars.append(char)
    
    processed_text = ''.join(chars)
    
    # Split by newlines and filter out empty commands
    commands = [cmd.strip() for cmd in processed_text.split('\n')]
    commands = [cmd for cmd in commands if cmd]
    
    # Filter out special marker lines that shouldn't be executed as commands
    commands = [cmd for cmd in commands if not (cmd.startswith(':TASK_COMPLETE:') or
                                              cmd.startswith(':INFO:') or
                                              cmd.startswith(':COMMAND:'))]
    
    logger.info(f"Parsed {len(commands)} commands from input")
    return commands

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in the given text.
    This is a rough approximation - about 4 characters per token.
    """
    return len(text) // 4

class LLMAgent:
    """Class to interact with the LLM via LiteLLM"""
    
    def __init__(self, config: Dict[str, Any], rag_retriever: RAGRetriever = None):
        """Initialize the LLM agent with the given configuration"""
        self.provider = config.get("provider", "ollama")
        self.model_name = config.get("model", "qwen2.5-coder:14b")
        self.api_base = config.get("api_base", "http://localhost:11434")
        
        # Fixed system prompt with all the technical instructions and special markers
        self.system_prompt = """You are an assistant controlling a Flipper Zero device through its CLI.
You need to intelligently determine when to provide commands versus information.

SPECIAL MARKERS - You MUST use ONE OR MORE of these exact formats in your responses:
1. When providing commands: ':COMMAND:' followed by the commands on separate lines.
   This should ONLY output commands to be sent to the device, with NOTHING else besides the marker and commands.

2. For informational responses: ':INFO: [your information here]'
   This provides information to the user and will be output to the terminal.

3. When needing clarification: ':QUESTION: [your question here]'
   This pauses execution and shows a prompt for user input.

4. When a task is fully complete: ':TASK_COMPLETE:'
   This returns to the main input prompt. You MUST include this when done to allow the user to respond to you.
   Without this marker, the system will continue looping indefinitely.

You can include MULTIPLE marker types in a single response. Your response is NOT limited to just commands.
For example, you can provide both :COMMAND: and :INFO: sections in the same response when appropriate.

The system processes your response in this order:
- First processes any :COMMAND: markers to execute commands
- Then processes any :INFO: markers to display information
- Finally processes :TASK_COMPLETE: if present, otherwise continues the loop with another prompt

IMPORTANT: If your response includes :COMMAND: or :INFO: but no :TASK_COMPLETE:, the system will
continue with another loop, feeding your previous output and context into the next input.
You must explicitly mark the task as complete with :TASK_COMPLETE: when all requested actions
are done, or the system will continue looping.

Examples:

For command execution only:
:COMMAND:
led g 255
led bl 255

For information requests only:
:INFO: The Flipper Zero has RGB LEDs that can display different colors.

For mixed command and information:
:COMMAND:
nfc detect
:INFO: Started the NFC detection process. The Flipper will now scan for nearby NFC tags.

For command with final task completion:
:COMMAND:
nfc save mycard
:TASK_COMPLETE:

Available commands include: info, gpio, ibutton, irda, lfrfid, nfc, subghz, usb,
vibro, update, bt, storage, bad_usb, backlight, led, and others.

IMPORTANT: The backlight command is 'led bl <value>' where value is 0-255."""

        # User configurable prompt that adds to the system prompt
        self.user_prompt = config.get("user_prompt", "")
        self.rag_retriever = rag_retriever
        
        # Initialize conversation history
        self.conversation_history = []
        self.max_history_tokens = config.get("max_history_tokens", 2000)
        self.max_recursion_depth = config.get("max_recursion_depth", 10)
        self.task_in_progress = False
        self.task_description = ""
        
        # Format the model name for LiteLLM
        if self.provider == "ollama":
            self.full_model = f"ollama/{self.model_name}"
        else:
            # Handle other providers if needed
            self.full_model = f"{self.provider}/{self.model_name}"
    
    def get_commands(self, user_input: str, previous_results: Optional[List[Tuple[str, str]]] = None) -> List[str]:
        """
        Get Flipper Zero commands from the LLM based on user input and previous results.
        
        Args:
            user_input: The original user query or task description
            previous_results: Optional list of previous command results as (command, response) tuples
            
        Returns:
            List of commands to execute or informational response
        """
        try:
            # Prepare API parameters
            api_params = {"api_base": self.api_base}
            
            # Retrieve relevant documentation if RAG is enabled
            context = ""
            if self.rag_retriever and self.rag_retriever.initialized:
                # Combine the user_input with any previous error context for better RAG
                search_query = user_input
                if previous_results:
                    # Add info about errors to the query to help find relevant docs
                    for cmd, resp in previous_results:
                        if "illegal option" in resp or "error" in resp.lower() or "usage:" in resp:
                            search_query += f" {cmd} error {resp}"
                
                docs = self.rag_retriever.retrieve(search_query)
                if docs:
                    context = "Here is relevant information about Flipper Zero CLI commands:\n\n"
                    context += "\n\n".join(docs)
                    logger.info(f"Retrieved {len(docs)} relevant documents")
                    logger.info(f"Retrieved {len(docs)} relevant documents")
            
            # Build the enhanced system prompt with user prompt and retrieved context
            enhanced_prompt = self.system_prompt
            
            # Add user prompt if specified
            if self.user_prompt:
                enhanced_prompt += f"\n\n{self.user_prompt}"
                
            # Add context if available
            if context:
                enhanced_prompt += "\n\n" + context
                
            # If this is a new task, store it
            if not self.task_in_progress and not previous_results:
                self.task_in_progress = True
                self.task_description = user_input
                logger.info(f"Starting new task: {user_input}")
            else:
                logger.info(f"Continuing task: {self.task_description}")
            
            # Prepare messages with conversation history
            messages = [{"role": "system", "content": enhanced_prompt}]
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add previous command results if any
            if previous_results:
                result_content = "Here are the results of the previous commands:\n\n"
                for cmd, resp in previous_results:
                    result_content += f"Command: {cmd}\nResponse: {resp}\n\n"
                
                result_content += "Based on these results, provide the next commands to execute, or use :TASK_COMPLETE: to indicate completion. Use :INFO: to provide information or analysis to the user."
                messages.append({"role": "user", "content": result_content})
                logger.info("Added previous command results to prompt")
            
            # Add the current user input if not just continuing from previous commands
            if not previous_results:
                messages.append({"role": "user", "content": user_input})
            
            logger.info(f"Sending request to {self.full_model}")
            print(f"{Colors.PURPLE}Thinking...{Colors.ENDC}")  # Keep yellow
            
            try:
                response = completion(
                    model=self.full_model,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more deterministic output
                    max_tokens=500,   # Allow for multiple commands
                    **api_params
                )
                
                # Extract the response content
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content
                else:
                    content = str(response)
                    
            except Exception as e:
                logger.error(f"LLM API error: {str(e)}", exc_info=True)
                return [f"INFO: Error communicating with LLM: {str(e)}"]
            
            logger.debug(f"Raw response: {content}")
            
            # Clean up the response
            content = content.strip()
            
            # Parse the response to extract different sections
            command_section = None
            info_section = None
            task_complete = False
            
            # Extract command section if present
            command_match = re.search(r':COMMAND:(.*?)(?=:(?:INFO|TASK_COMPLETE):|$)', content, re.DOTALL)
            if command_match:
                command_section = command_match.group(1).strip()
                logger.info(f"Found command section: {command_section}")
            
            # Extract info section if present
            info_match = re.search(r':INFO:\s*(.*?)(?=:(?:COMMAND|TASK_COMPLETE):|$)', content, re.DOTALL)
            if info_match:
                info_section = info_match.group(1).strip()
                logger.info(f"Found info section: {info_section}")
            
            # Check if task is complete
            task_complete = ':TASK_COMPLETE:' in content
            if task_complete:
                logger.info("Task marked as complete by AI")
                self.task_in_progress = False
                self.task_description = ""
                # We'll let the process_user_request function handle the final summary
            
            # Process command section if present
            commands = []
            if command_section:
                # Clean the command section
                # Remove any markdown code block markers
                cleaned_command = re.sub(r'```.*?\n', '', command_section, flags=re.DOTALL)
                cleaned_command = re.sub(r'```', '', cleaned_command)
                
                # Remove any "Command:" prefixes
                cleaned_command = re.sub(r'^.*?Command[s]?:[ \t]*', '', cleaned_command, flags=re.MULTILINE)
                
                # Final cleanup and strip
                cleaned_command = cleaned_command.strip()
                
                # Parse into individual commands
                if cleaned_command:
                    commands = parse_commands(cleaned_command)
                    logger.info(f"Generated {len(commands)} commands: {commands}")
                else:
                    logger.info("Command section was empty after cleaning")
            
            # Process info section if present (return as a special command that will be handled)
            if info_section and not commands:
                logger.info(f"Information response: {info_section}")
                return [f"INFO: {info_section}"]
                
            # Final task completion check
            if task_complete:
                # Always return the task completion marker to ensure process_user_request knows it's complete
                # If there are commands, include them along with the completion marker
                if commands:
                    commands.append("INFO: Task completed successfully.")
                    return commands
                else:
                    return ["INFO: Task completed successfully."]
            
            logger.info(f"Generated {len(commands)} commands: {commands}")
            
            # Add to conversation history
            if not previous_results:  # Only add the original user query once
                self.conversation_history.append({"role": "user", "content": user_input})
            
            # Store conversation history
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Prune history if it exceeds the token limit
            self._prune_history_by_tokens()
            
            return commands
        except Exception as e:
            error_msg = f"Error generating commands: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [f"INFO: {error_msg}"]
    
    def _prune_history_by_tokens(self):
        """Prune conversation history to stay under the token limit"""
        if not self.conversation_history:
            return
            
        # Calculate current token count
        total_tokens = 0
        for message in self.conversation_history:
            total_tokens += estimate_tokens(message["content"])
            
        # If under limit, no action needed
        if total_tokens <= self.max_history_tokens:
            return
        
        # Log before pruning
        logger.info(f"History token count ({total_tokens}) exceeds limit ({self.max_history_tokens}), pruning...")
            
        # Remove oldest messages until under token limit
        while total_tokens > self.max_history_tokens and len(self.conversation_history) > 2:
            # Remove oldest exchange (user + assistant message)
            removed_user = self.conversation_history.pop(0)
            removed_assistant = self.conversation_history.pop(0)
            
            # Recalculate token count
            total_tokens -= estimate_tokens(removed_user["content"])
            total_tokens -= estimate_tokens(removed_assistant["content"])
            
        logger.info(f"After pruning: {len(self.conversation_history)} messages, ~{total_tokens} tokens")
    
    def generate_summary(self) -> str:
        """
        Generate a summary of the conversation so far.
        This is called by the Python script rather than relying on the LLM to provide summaries.
        Called automatically on TASK_COMPLETE or after N iterations.
        
        Returns:
            Summary string
        """
        # Check if there's meaningful content to summarize
        if not self.conversation_history:
            return "Task just started, no progress to summarize yet."
            
        # Get the last few exchanges for context
        recent_history = self.conversation_history[-8:] if len(self.conversation_history) >= 8 else self.conversation_history
        
        # Format the conversation for summarization
        summary_prompt = "Please provide a brief summary of what has been accomplished so far:\n\n"
        
        # Add the recent history
        for message in recent_history:
            role = message["role"]
            content = message["content"]
            summary_prompt += f"{role.upper()}: {content}\n\n"
            
        # Add the request for summary
        summary_prompt += "Summary of progress so far:"
        
        try:
            # Use the same LLM to generate a summary
            response = completion(
                model=self.full_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the progress of the task so far in a concise paragraph."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=150,
                api_base=self.api_base
            )
            
            # Extract the response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                summary = response.choices[0].message.content
            else:
                summary = str(response)
                
            logger.info(f"Generated summary: {summary}")
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Unable to generate summary: {str(e)}"
            
    def add_execution_results_to_history(self, results: List[Tuple[str, str]]):
        """
        Add command execution results to the conversation history.
        This ensures the summary generation has access to what commands were run
        and what their results were.
        
        Args:
            results: List of (command, response) tuples from command execution
        """
        if not results:
            return
            
        # Format the results into a readable message
        result_content = "Command execution results:\n\n"
        for cmd, resp in results:
            result_content += f"Command: {cmd}\nResponse: {resp}\n\n"
        
        # Add as a system message to conversation history
        self.conversation_history.append({"role": "system", "content": result_content})
        logger.info("Added execution results to conversation history for summarization")

def process_user_request(user_input: str, flipper_agent: FlipperZeroManager, llm_agent: LLMAgent, recursion_depth: int = 0):
    """
    Process a user request and execute any commands.
    
    Args:
        user_input: The user's input text
        flipper_agent: The agent that communicates with Flipper Zero
        llm_agent: The agent that generates commands using LLM
        recursion_depth: Current recursion depth for follow-up commands
        
    Returns:
        None
    """
    # Guard against excessive recursion
    if recursion_depth >= llm_agent.max_recursion_depth:
        logger.warning(f"Maximum recursion depth reached ({llm_agent.max_recursion_depth}), stopping chain")
        print(f"\n{Colors.FAIL}Maximum loop depth reached ({llm_agent.max_recursion_depth}). Please issue a new command to continue.{Colors.ENDC}")
        # Generate a final summary when we stop due to recursion limit
        summary = llm_agent.generate_summary()
        print(f"\n{Colors.CYAN}Final Summary:{Colors.ENDC}")
        print(f"{summary}\n")
        return
    
    # Generate summary every 3 iterations to keep context fresh
    if recursion_depth > 0 and recursion_depth % 3 == 0:
        summary = llm_agent.generate_summary()
        print(f"\n{Colors.CYAN}Progress Summary:{Colors.ENDC}")
        print(f"{summary}\n")
    
    # Get commands from LLM
    response = llm_agent.get_commands(user_input)
    
    # Handle empty response
    if not response:
        logger.warning("No response was generated by the LLM")
        print(f"\n{Colors.WARNING}No response was generated. Try rephrasing your request.{Colors.ENDC}")
        return
    
    # Check if this response contains a task completion marker
    task_completed = any(cmd.startswith("INFO: Task completed successfully") for cmd in response)
    
    # Extract special command types
    info_commands = [cmd[5:].strip() for cmd in response if cmd.startswith("INFO:")]
    question_commands = [cmd[10:].strip() for cmd in response if cmd.startswith("QUESTION:")]
    regular_commands = [cmd for cmd in response if not cmd.startswith("INFO:") and not cmd.startswith("QUESTION:")]
    
    # Process questions first
    for question in question_commands:
        print(f"{Colors.BLUE}{'─' * 50}{Colors.ENDC}")
        print(f"{Colors.BLUE}? {question}{Colors.ENDC}")
        print(f"{Colors.BLUE}{'─' * 50}{Colors.ENDC}")
        return  # Pause execution for user input
    
    # Process informational responses
    for info in info_commands:
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
        print(f"{Colors.GREEN}# Information:{Colors.ENDC}")
        print(f"{Colors.BOLD}{info}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
    
    # Process commands after handling questions/info
    
    # Process regular commands if any
    results = []
    if regular_commands:
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
        print(f"{Colors.PURPLE}Executing {len(regular_commands)} commands...{Colors.ENDC}")
        for i, cmd in enumerate(regular_commands, 1):
            print(f"{Colors.BOLD}{i}.{Colors.ENDC} {cmd}")
        # Add a green separator line for clarity
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
        
        # Execute commands and get results
        results = flipper_agent.execute_commands(regular_commands)
        
        # Add results to conversation history for better summaries
        llm_agent.add_execution_results_to_history(results)
    
    # NOW check if task is complete (either by marker or directly detected)
    # This ensures task completion is the last thing shown
    if task_completed or not llm_agent.task_in_progress:
        # Generate final summary if task is complete
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
        summary = llm_agent.generate_summary()
        print(f"{Colors.GREEN}# Final Summary:{Colors.ENDC}")
        print(f"{summary}")
        # Make "Task completed" the last thing displayed before a new prompt
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
        print(f"{Colors.GREEN}✓ Task completed {Colors.ENDC}")
        print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
        return
    
    # Default behavior: continue the loop if not complete and under recursion limit
    if recursion_depth < llm_agent.max_recursion_depth:
        # Always analyze results, even if no commands were executed
        print(f"\n{Colors.PURPLE}Analyzing results...{Colors.ENDC}")
        
        # Pass the results explicitly so the LLM can analyze them
        if results:
            # Get next set of commands based on the results
            process_user_request(user_input, flipper_agent, llm_agent, recursion_depth + 1)
        else:
            # No commands were executed, just continue with the request
            process_user_request(user_input, flipper_agent, llm_agent, recursion_depth + 1)

# === Main Execution Flow ===
def main():
    """Orchestrate the application startup and shutdown"""
    args = parse_arguments()
    config = load_configuration(args)
    rag_retriever = initialize_rag_system(args)
    flipper_agent = establish_flipper_connection(config)
    llm_agent = configure_llm_agent(config, rag_retriever, args)
    display_connection_banner(config, llm_agent)
    run_interactive_loop(flipper_agent, llm_agent)

# === Configuration Management ===
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

# === Hardware Integration ===
def establish_flipper_connection(config: Dict[str, Any]) -> FlipperZeroManager:
    """Initialize and validate Flipper Zero connection"""
    agent = FlipperZeroManager(port=config["flipper"]["port"])
    if not agent.connect():
        logger.error("Flipper Zero connection failed")
        print(f"{Colors.FAIL}Connection failed - check device and permissions{Colors.ENDC}")
        sys.exit(1)
    return agent

# === AI Components Setup ===
def initialize_rag_system(args: argparse.Namespace) -> Optional[RAGRetriever]:
    """Configure the RAG retrieval system if enabled"""
    if args.no_rag:
        logger.info("RAG system disabled via command line")
        return None
    
    print(f"{Colors.PURPLE}Initializing RAG knowledge base...{Colors.ENDC}")
    rag = RAGRetriever()
    return rag if rag.initialize() else None

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

# === User Interface ===
def display_connection_banner(config: Dict[str, Any], agent: LLMAgent):
    """Show startup connection status and configuration"""
    print(f"{Colors.GREEN}{'─' * 58}")
    print(f"✓ Connected to Flipper Zero on {config['flipper']['port']:^35}")
    print(f"✓ LLM Provider/Model: {agent.provider}/{agent.model_name:^35}")
    print(f"✓ Context History: {agent.max_history_tokens} tokens{'':^20}")
    print(f"✓ Log File: {LOG_FILE:^45}")
    print(f"{'─' * 58}┘{Colors.ENDC}")
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

if __name__ == "__main__":
    main()