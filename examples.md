# Flipper Zero AI Agent Examples (PyFlipper)

This document provides examples of natural language commands you can use with the Flipper Zero AI Agent and the expected Flipper Zero CLI commands they generate. These commands are sent to the Flipper Zero using the PyFlipper library.

## Basic Device Commands

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Show device information" | `info` |
| "Display system stats" | `system_info` |
| "What's the battery level?" | `battery info` |
| "Show available storage" | `storage info` |
| "Reboot the device" | `system reboot` |
| "Show running apps" | `ps` |

## GPIO Controls

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Set GPIO pin 5 to high" | `gpio set pc5 1` |
| "Turn on GPIO pin 3" | `gpio set pc3 1` |
| "Set all GPIO pins to low" | `gpio set all 0` |
| "Read the value of GPIO pin 2" | `gpio read pc2` |
| "Show GPIO pin status" | `gpio list` |

## NFC Operations

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Scan for NFC tags" | `nfc detect` |
| "Read an NFC tag" | `nfc read` |
| "Save the current NFC data" | `nfc save` |
| "Emulate the saved NFC tag" | `nfc emulate` |
| "List saved NFC tags" | `storage list /ext/nfc` |

## RFID Operations

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Scan for RFID cards" | `lfrfid read` |
| "Emulate an RFID card" | `lfrfid emulate` |
| "Write RFID data" | `lfrfid write` |
| "List saved RFID cards" | `storage list /ext/lfrfid` |

## Infrared Controls

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Capture an IR signal" | `ir rx` |
| "Transmit the saved IR signal" | `ir tx` |
| "List saved IR signals" | `storage list /ext/infrared` |

## Utility Commands

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Show current date and time" | `date` |
| "Turn on the LED" | `led g on` |
| "Make the device vibrate" | `vibro 100` |
| "Show help for bluetooth commands" | `bt help` |
| "List files on the SD card" | `storage list /ext` |

## PyFlipper API Examples

You can also use more natural language that leverages PyFlipper's structured API:

| Natural Language Command | Behind-the-scenes Implementation |
|--------------------------|----------------------------------|
| "Change the green LED color" | Uses PyFlipper's LED API |
| "Show storage information" | Uses PyFlipper's Storage API |
| "Get battery percentage" | Uses PyFlipper's Power API |
| "Scan for Bluetooth devices nearby" | Uses PyFlipper's BT API |

## Context-Aware Conversation Examples

The agent now maintains conversation context, allowing for more natural follow-up commands that reference previous interactions:

| Conversation Flow | Command Sequence |
|------------------|------------------|
| "Turn on the green LED" <br> "Make it brighter" | `led g on` <br> `led g 100` |
| "Turn on the backlight" <br> "Set it to 50%" | `backlight on` <br> `backlight 50` |
| "Turn on the backlight" <br> "Now turn it off" | `backlight on` <br> `backlight off` |
| "Start scanning for NFC" <br> "Stop scanning" | `nfc detect` <br> `nfc stop` |
| "Show battery level" <br> "Is it charging?" | `battery info` <br> `battery status` |

## Information Requests

The agent can also respond to information requests without executing commands:

| Information Request | Response |
|---------------------|----------|
| "How do I scan NFC tags?" | `INFO: To scan NFC tags with Flipper Zero, use the 'nfc detect' command...` |
| "Explain the LED colors" | `INFO: The Flipper Zero has RGB LEDs that can display different colors...` |
| "What commands are available for bluetooth?" | `INFO: Bluetooth commands include: bt info, bt on, bt off, bt scan...` |

## Complex Examples

| Natural Language Command | Generated CLI Command |
|--------------------------|----------------------|
| "Scan for NFC tags, save the data, and then emulate it" | `nfc detect` <br> `nfc save mycard` <br> `nfc emulate mycard` |
| "Check battery level and then turn on the green LED if it's above 50%" | `battery info` <br> `led g on` |
| "Scan for bluetooth devices and save the results to a file" | `bt scan` <br> `storage write /ext/bt_scan.txt` |

## Tips for Effective Commands

1. **Be specific** about what you want the Flipper Zero to do
2. **Use clear action verbs** (scan, read, write, display, show)
3. **Specify targets** when applicable (which pin, which LED color)
4. **Use technical terms** when you know them (NFC, RFID, GPIO)

Remember that complex operations might need to be broken down into multiple commands.

## Context Awareness Tips

1. **Use natural references** to previous commands (e.g., "it", "that", "the light")
2. **Be specific when needed** to avoid ambiguity
3. **Remember the agent maintains state awareness** between commands
4. **For information-only requests**, ask questions naturally (e.g., "How do I...?" "Explain...")