"""
Class to interact with the LLM via LiteLLM
"""

import logging
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple

from litellm import completion
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from pathlib import Path

from colors import Colors
from rag_retriever import RAGRetriever

logger = logging.getLogger("FlipperAgent")
ai_logger = logging.getLogger("FlipperAgentAI")

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

        # Fixed system prompt with explicit completion requirements
        self.system_prompt = """You are an assistant controlling a Flipper Zero device through its CLI.
You need to intelligently determine when to provide commands versus information.

Use the following tools to structure your responses:

1. execute_commands: For sending commands to the Flipper Zero device
2. provide_information: For displaying information to the user
3. ask_question: For asking the user for clarification
4. mark_task_complete: To indicate the task is finished

You MUST use these tools to structure your responses. Do not use any special markers.

The system will automatically process tool calls in the order they are provided.

CRITICAL: After completing the user's request, you MUST call mark_task_complete to return control to the user.
Without this, the system will loop indefinitely.

IMPORTANT: For every command execution, you MUST include mark_task_complete in the same response.
The only exceptions are when you need to ask a question or provide additional information.

RULES:
1. When you have COMPLETELY satisfied the user's request, ALWAYS include mark_task_complete
2. If you need to execute multiple commands to satisfy the request, do so before marking complete
3. Only ask questions when you need clarification to complete the task
4. Provide information when it helps the user understand what was done

Examples:

For command execution with completion:
[
  {
    "name": "execute_commands",
    "arguments": {
      "commands": ["led bl 0"]
    }
  },
  {
    "name": "mark_task_complete",
    "arguments": {}
  }
]

For information display with completion:
[
  {
    "name": "provide_information",
    "arguments": {
      "information": "Backlight turned off"
    }
  },
  {
    "name": "mark_task_complete",
    "arguments": {}
  }
]

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

    def get_commands(self, user_input: str, previous_results: Optional[List[Tuple[str, str]]] = None) -> List[dict]:
        """
        Get Flipper Zero commands from the LLM based on user input and previous results.

        Args:
            user_input: The original user query or task description
            previous_results: Optional list of previous command results as (command, response) tuples

        Returns:
            List of tool call objects (each is a dict with 'name' and 'arguments')
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

                # Add context for next action
                if any("error" in resp.lower() for cmd, resp in previous_results):
                    result_content += "Based on these results, provide the next tool calls to continue the task."
                else:
                    result_content += "The command executed successfully. You MUST include mark_task_complete in your response to prevent an infinite loop."

                messages.append({"role": "user", "content": result_content})
                logger.info("Added previous command results to prompt")

            # Add the current user input if not just continuing from previous commands
            if not previous_results:
                messages.append({"role": "user", "content": user_input})

            logger.info(f"Sending request to {self.full_model}")
            print(f"{Colors.PURPLE}Thinking...{Colors.ENDC}")  # Keep yellow

            try:
                # Use function calling with tool definitions
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_commands",
                            "description": "Execute commands on the Flipper Zero device",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "commands": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of commands to execute"
                                    }
                                },
                                "required": ["commands"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "provide_information",
                            "description": "Provide information to the user",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "information": {
                                        "type": "string",
                                        "description": "Information to display to the user"
                                    }
                                },
                                "required": ["information"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "ask_question",
                            "description": "Ask the user a question",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "Question to ask the user"
                                    }
                                },
                                "required": ["question"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "mark_task_complete",
                            "description": "Mark the task as complete",
                            "parameters": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    }
                ]

                response = completion(
                    model=self.full_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=500,
                    **api_params
                )

                # Extract tool calls from response
                tool_calls = []
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    message = response.choices[0].message
                    if hasattr(message, 'tool_calls'):
                        for call in message.tool_calls:
                            try:
                                # Parse arguments as JSON
                                arguments = json.loads(call.function.arguments)
                                tool_calls.append({
                                    "name": call.function.name,
                                    "arguments": arguments
                                })
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")
                                ai_logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")

                logger.info(f"Generated {len(tool_calls)} tool calls")

                # Log AI response for debugging
                ai_logger.info("===== AI RESPONSE =====")
                ai_logger.info(f"Model: {self.full_model}")
                ai_logger.info(f"Input Messages: {json.dumps(messages, indent=2)}")
                ai_logger.info(f"Raw Response: {response}")
                ai_logger.info(f"Tool Calls: {json.dumps(tool_calls, indent=2)}")
                ai_logger.info("======================")

                # Validate tool calls format
                if tool_calls:
                    # Check if mark_task_complete is missing after execute_commands
                    has_execute = any(call['name'] == 'execute_commands' for call in tool_calls)
                    has_mark = any(call['name'] == 'mark_task_complete' for call in tool_calls)

                    if has_execute and not has_mark:
                        # Add mark_task_complete if missing
                        tool_calls.append({
                            "name": "mark_task_complete",
                            "arguments": {}
                        })
                        logger.warning("Added missing mark_task_complete tool call")

                return tool_calls

            except Exception as e:
                error_msg = f"LLM API error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                ai_logger.error(f"LLM API error: {str(e)}", exc_info=True)
                return [{"name": "provide_information", "arguments": {"information": f"Error: {str(e)}"}}]

        except Exception as e:
            error_msg = f"Error generating tool calls: {str(e)}"
            logger.error(error_msg, exc_info=True)
            ai_logger.error(error_msg, exc_info=True)
            return [{"name": "provide_information", "arguments": {"information": error_msg}}]

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