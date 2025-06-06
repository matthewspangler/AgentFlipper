import sys
import os

# Path setup for PyFlipper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "pyFlipper/src"))

# Import PyFlipper
try:
    from pyflipper.pyflipper import PyFlipper
except ImportError:
    # Handle import error if necessary, though the main script also does this
    pass

from colors import Colors
"""
Base class for hardware device management
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("FlipperAgent")

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
            # Display command in orange color with separator
            print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ Sent command -> {command}{Colors.ENDC}")
            print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
            logger.info(f"Executing command: {command}")

            # Access the private serial wrapper to send arbitrary commands
            response = self.connection._serial_wrapper.send(command)
            logger.info(f"Command response: {response}")

            # Handle empty responses and device prompts
            cleaned_response = response.strip()
            if not cleaned_response or cleaned_response in [">", ">:"]:
                return "Command executed successfully"
            return response
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