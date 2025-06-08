import sys
import os
import logging
from typing import Dict, Any, Optional, List, Tuple

# Path setup for PyFlipper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "pyFlipper/src"))

# Import PyFlipper
try:
    from pyflipper.pyflipper import PyFlipper
except ImportError:
    # Handle import error if necessary, though the main script also does this
    pass

"""
Base class for hardware device management
"""

logger = logging.getLogger("FlipperAgent")

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
            # Assuming PyFlipper constructor is synchronous
            self.connection = PyFlipper(com=self.port)
            # Assuming device_info.info() is synchronous or handled internally by PyFlipper connect
            self.connection.device_info.info()  # Test connection
            self._connected = True
            logger.info("Flipper Zero connection established")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.disconnect()
            return False

    async def send_command(self, command: str) -> str:
        """Execute a command on the Flipper Zero device - Now async"""
        if not self.is_connected:
            raise ConnectionError("Device not connected")

        try:
            # Display command in orange color with separator
            # Using direct print here, as this is a lower-level command sending function
            # UI updates will be handled in execute_commands
            # print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
            # print(f"{Colors.GREEN}✓ Sent command -> {command}{Colors.ENDC}")
            # print(f"{Colors.GREEN}{'─' * 50}{Colors.ENDC}")
            logger.info(f"Executing command: {command}")

            # Access the private serial wrapper to send arbitrary commands
            # This is the actual async call
            # Assuming self.connection._serial_wrapper.send is awaitable
            response = self.connection._serial_wrapper.send(command) # Removed await - check for blocking
            logger.info(f"Command response: {response}")

            # Handle empty responses and device prompts
            cleaned_response = response.strip()
            if not cleaned_response or cleaned_response in [">", ">:"]:
                return "Command executed successfully"
            return response
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg)
            # Re-raise the exception after logging, so execute_commands can catch it
            raise

    async def execute_commands(self, commands: List[str], app_instance) -> List[Tuple[str, str]]:
        """
        Execute multiple commands in sequence and return a list of (command, response) tuples.
        This allows tracking what commands were executed and their responses.
        Args:
            commands: List of commands to execute
            app_instance: The TextualApp instance for UI updates.
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

            try:
                # Display sent command using the app_instance
                # Use fg-orange class for command messages
                app_instance.display_message(f"{'─' * 79}")
                app_instance.display_message(f"[class='fg-orange']✓ Sent command -> {cmd}[/class]")
                app_instance.display_message(f"{'─' * 79}")


                # Execute command and get response - AWAITING the async send_command
                response = await self.send_command(cmd)
                results.append((cmd, response))

                # Display the response with clearer formatting using the app_instance
                # Use fg-orange class for response messages
                app_instance.display_message(f"{'─' * 79}")
                app_instance.display_message(f"[class='fg-orange']# Device Response:[/class]") # Removed erroneous comment
                app_instance.display_message(f"[class='fg-orange']{response}[/class]")

                # Add single separator between agent actions using the app_instance
                app_instance.display_message(f"{'─' * 79}")

            except Exception as e:
                # Catch exceptions from send_command and record the error
                error_msg = f"Error executing command '{cmd}': {str(e)}"
                logger.error(error_msg)
                results.append((cmd, f"ERROR: {error_msg}"))
                # Keep error messages red for visibility
                app_instance.display_message(f"[red]{'─' * 79}[/red]")
                app_instance.display_message(f"[red]# Device Response:[/red]")
                app_instance.display_message(f"[bold red]ERROR: {error_msg}[/bold red]")
                app_instance.display_message(f"[red]{'─' * 79}[/red]")


        logger.info(f"Executed {len(results)} commands")
        return results

    def disconnect(self):
        """Disconnect from the Flipper Zero"""
        # PyFlipper doesn't have an explicit disconnect method,
        # but the serial port will be closed when the object is deleted
        self.flipper = None
        logger.info("Disconnected from Flipper Zero")