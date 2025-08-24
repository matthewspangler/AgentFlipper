import sys
import os
import logging
import time
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

logger = logging.getLogger(__name__)

from ui.colors import Colors


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
                logger.info("Disconnecting hardware connection...")
                self.connection.close()
                logger.debug("Hardware connection closed gracefully")
            except AttributeError as e:
                logger.warning(f"Connection object missing close() method: {str(e)}")
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}", exc_info=True)
        else:
            logger.debug("No connection object to disconnect")
            
        # Always reset state regardless of exceptions
        self.connection = None
        self._connected = False
        logger.info("Hardware connection terminated and state reset")

class FlipperZeroManager(HardwareManager):
    """Manages Flipper Zero device communication using PyFlipper"""

    def connect(self) -> bool:
        """Establish connection to Flipper Zero"""
        try:
            logger.info(f"Attempting to connect to Flipper Zero on port {self.port}")
            
            # Check if port exists
            if not os.path.exists(self.port):
                logger.error(f"Port {self.port} does not exist or is not accessible")
                return False
                
            # Assuming PyFlipper constructor is synchronous
            self.connection = PyFlipper(com=self.port)
            
            # Test connection by getting device info
            logger.debug("Testing connection by querying device info...")
            info_result = self.connection.device_info.info()  # Test connection
            logger.debug(f"Connection test result: {info_result}")
            
            self._connected = True
            logger.info(f"Flipper Zero connection successfully established on {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Flipper Zero connection failed: {str(e)}", exc_info=True)
            self.disconnect()
            return False

    async def send_command(self, command: str) -> str:
        """Execute a command on the Flipper Zero device - Now async"""
        # Enhanced connection check with more detailed error
        if not self.is_connected:
            logger.error(f"Attempted to send command '{command}' but device is not connected")
            raise ConnectionError(f"Device not connected. Please ensure the Flipper Zero is connected at {self.port} and the connection has been established.")
        
        if not self.connection:
            logger.error(f"Connection object is None when trying to send command '{command}'")
            raise ConnectionError(f"Connection object is None. Device may have been disconnected.")

        try:
            # Log the command being sent with debug info
            logger.info(f"Sending command to Flipper Zero: '{command}'")
            logger.debug(f"Connection status: {self.is_connected}, Port: {self.port}")

            # Access the private serial wrapper to send arbitrary commands
            response = self.connection._serial_wrapper.send(command)
            logger.info(f"Command response: {response}")

            # Handle empty responses and device prompts
            cleaned_response = response.strip()
            if not cleaned_response or cleaned_response in [">", ">:"]:
                return "Command executed successfully"
            return response
            
        except AttributeError as e:
            error_msg = f"Connection appears broken (missing _serial_wrapper): {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._connected = False  # Mark as disconnected since connection is broken
            raise ConnectionError(f"Connection to Flipper Zero is broken: {str(e)}")
            
        except Exception as e:
            error_msg = f"Error executing command '{command}': {str(e)}"
            logger.error(error_msg, exc_info=True)
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
        
        # Check connection status before attempting to execute commands
        if not self.is_connected:
            try:
                # Attempt to reconnect
                app_instance.display_message(f"[yellow]Device connection lost. Attempting to reconnect...[/yellow]")
                reconnect_success = self.connect()
                
                if reconnect_success:
                    app_instance.display_message(f"[green]Successfully reconnected to device[/green]")
                else:
                    error_msg = f"Failed to reconnect to device on port {self.port}"
                    app_instance.display_message(f"[bold red]ERROR: {error_msg}[/bold red]")
                    # Return error for all commands
                    for cmd in commands:
                        if cmd.strip() and not cmd.startswith("INFO:") and not cmd.startswith(":"):
                            results.append((cmd, f"ERROR: Device not connected and reconnection failed"))
                    return results
            except Exception as e:
                app_instance.display_message(f"[bold red]ERROR: Reconnection attempt failed: {str(e)}[/bold red]")
                # Return error for all commands
                for cmd in commands:
                    if cmd.strip() and not cmd.startswith("INFO:") and not cmd.startswith(":"):
                        results.append((cmd, f"ERROR: Device not connected and reconnection failed: {str(e)}"))
                return results

        for cmd in commands:
            # Skip empty commands or special markers
            cmd = cmd.strip()
            if not cmd or cmd.startswith("INFO:") or cmd.startswith(":"):
                logger.warning(f"Skipping invalid command: {cmd}")
                continue

            try:
                # Display sent command using the app_instance
                app_instance.display_message(f"{'─' * 79}")
                app_instance.display_message(f"[class='fg-orange']✓ Sent command -> {cmd}[/class]")
                app_instance.display_message(f"{'─' * 79}")

                # Execute command and get response
                response = await self.send_command(cmd)
                results.append((cmd, response))

                # Display the response with clearer formatting
                app_instance.display_message(f"{'─' * 79}")
                app_instance.display_message(f"[class='fg-orange']# Device Response:[/class]")
                app_instance.display_message(f"[class='fg-orange']{response}[/class]")
                app_instance.display_message(f"{'─' * 79}")

            except ConnectionError as e:
                # Handle connection errors specially - attempt to reconnect
                error_msg = f"Connection error executing command '{cmd}': {str(e)}"
                logger.error(error_msg)
                
                # Attempt to reconnect once
                app_instance.display_message(f"[yellow]Connection lost. Attempting to reconnect...[/yellow]")
                try:
                    reconnect_success = self.connect()
                    if reconnect_success:
                        app_instance.display_message(f"[green]Successfully reconnected. Retrying command...[/green]")
                        # Retry the command once after successful reconnection
                        try:
                            response = await self.send_command(cmd)
                            results.append((cmd, response))
                            
                            # Display the retry response
                            app_instance.display_message(f"{'─' * 79}")
                            app_instance.display_message(f"[green]# Device Response (after reconnection):[/green]")
                            app_instance.display_message(f"[green]{response}[/green]")
                            app_instance.display_message(f"{'─' * 79}")
                            
                            # Skip the error reporting since we recovered
                            continue
                        except Exception as retry_e:
                            # If retry fails, fall through to error handling below
                            error_msg = f"Command retry failed: {str(retry_e)}"
                            logger.error(error_msg)
                    else:
                        app_instance.display_message(f"[red]Reconnection failed[/red]")
                except Exception as reconnect_e:
                    app_instance.display_message(f"[red]Reconnection attempt failed: {str(reconnect_e)}[/red]")
                
                # Record the error in results
                results.append((cmd, f"ERROR: {error_msg}"))
                
                # Display error in UI
                app_instance.display_message(f"[red]{'─' * 79}[/red]")
                app_instance.display_message(f"[red]# Device Response:[/red]")
                app_instance.display_message(f"[bold red]ERROR: {error_msg}[/bold red]")
                app_instance.display_message(f"[red]{'─' * 79}[/red]")
                
            except Exception as e:
                # Handle other exceptions
                error_msg = f"Error executing command '{cmd}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                results.append((cmd, f"ERROR: {error_msg}"))
                
                # Display error in UI
                app_instance.display_message(f"[red]{'─' * 79}[/red]")
                app_instance.display_message(f"[red]# Device Response:[/red]")
                app_instance.display_message(f"[bold red]ERROR: {error_msg}[/bold bold]")
                app_instance.display_message(f"[red]{'─' * 79}[/red]")

        logger.info(f"Executed {len(results)} commands with {sum(1 for _, r in results if 'ERROR:' in r)} errors")
        return results

    def disconnect(self):
        """Disconnect from the Flipper Zero"""
        # Call parent class disconnect method to handle proper cleanup
        super().disconnect()
        
        # Additional PyFlipper-specific cleanup if needed
        logger.info("Disconnected from Flipper Zero")