#!/usr/bin/env python3
"""
Documentation Loader for Flipper Zero CLI

This script fetches and processes the Flipper Zero CLI documentation
to be used in a RAG system powered by LangChain.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# URL of the Flipper Zero CLI documentation
FLIPPER_CLI_DOC_URL = "https://docs.flipper.net/development/cli/"
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CLI_DOCS_PATH = os.path.join(DOCS_DIR, "flipper_cli_docs.txt")
VECTOR_STORE_PATH = os.path.join(DOCS_DIR, "flipper_cli_faiss")

def fetch_documentation(url: str = FLIPPER_CLI_DOC_URL) -> str:
    """Fetch the CLI documentation from the Flipper Zero website"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"{Colors.FAIL}Error fetching documentation: {str(e)}{Colors.ENDC}")
        return ""

def process_page_content(html_content: str) -> str:
    """Extract and process the CLI commands and their descriptions"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the main content
    main_content = soup.find('div', class_='markdown-body') or soup.find('article') or soup.body
    
    if not main_content:
        return "Failed to extract content from the documentation page."
    
    # Extract text from the main content
    content_text = main_content.get_text(separator='\n\n', strip=True)
    
    # Add supplementary CLI command descriptions
    command_descriptions = """
# Flipper Zero CLI Command Reference

## Core Commands

### info
Display device information including firmware version, hardware model, and other system details.

### gpio
Control the GPIO pins on the Flipper Zero.
- gpio set [pin] [0|1|?] - Set a GPIO pin high (1), low (0), or show state (?)
- gpio read [pin] - Read the current state of a GPIO pin

### ibutton
Work with iButton devices.
- ibutton read - Read an iButton device
- ibutton write - Write data to an iButton device
- ibutton emulate - Emulate an iButton device

### irda (ir)
Work with infrared signals.
- ir rx - Receive IR signals
- ir tx - Transmit IR signals
- ir set [protocol] - Set IR protocol

### lfrfid
Low frequency RFID operations.
- lfrfid read - Read RFID card
- lfrfid write - Write data to RFID card
- lfrfid emulate - Emulate RFID card
- lfrfid detect - Detect RFID cards

### nfc
Work with NFC tags.
- nfc detect - Detect NFC tags
- nfc read - Read NFC tag data
- nfc emulate - Emulate NFC tag
- nfc save - Save current NFC data

### subghz
Sub-GHz operations.
- subghz rx - Receive Sub-GHz signals
- subghz tx - Transmit Sub-GHz signals
- subghz read - Read saved Sub-GHz files
- subghz write - Write Sub-GHz data to file

## System Commands

### led
Control LEDs.
- led r [on|off] - Control red LED
- led g [on|off] - Control green LED
- led b [on|off] - Control blue LED
- led bl [on|off] - Control backlight

### vibro
Control the vibration motor.
- vibro [0-100] - Set vibration strength (0-100%)

### usb
USB mode controls.
- usb start [mode] - Start USB in specific mode
- usb stop - Stop USB mode

### bt
Bluetooth operations.
- bt on - Turn Bluetooth on
- bt off - Turn Bluetooth off
- bt scan - Scan for Bluetooth devices
- bt forget - Forget paired devices

### storage
File system operations.
- storage info - Show storage information
- storage list [path] - List files in a directory
- storage stat [path] - Show file statistics
- storage write [path] [data] - Write data to a file
- storage remove [path] - Remove a file
- storage md5 [path] - Calculate MD5 hash of a file

### update
Update firmware-related commands.
- update [options] - Update firmware

### ps
Show running processes/applications.

### free
Display memory usage information.

### help
Display help information about available commands.

### date
Show or set the current date and time.

### system
System-related commands.
- system reboot - Reboot the device
- system info - Show system information
- system heap - Show heap memory usage
    """
    
    # Combine the extracted content with our supplementary information
    full_content = content_text + "\n\n" + command_descriptions
    
    return full_content

def create_vector_store(text_content: str) -> None:
    """Create a vector store from the documentation content"""
    # Ensure docs directory exists
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # Save the text content to a file
    with open(CLI_DOCS_PATH, 'w') as f:
        f.write(text_content)
    
    # Load the document
    loader = TextLoader(CLI_DOCS_PATH)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split documentation into {len(chunks)} chunks")
    
    # Initialize the embedding model - using a small model that can run locally
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create and save the vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    print(f"Vector store created and saved to {VECTOR_STORE_PATH}")

def main():
    """Main function"""
    print(f"Fetching documentation from {FLIPPER_CLI_DOC_URL}...")
    html_content = fetch_documentation()
    
    if not html_content:
        print("Failed to fetch documentation. Using backup data...")
        # In a real scenario, we would have a backup data source
        # For now, we'll proceed with creating dummy data
    
    print("Processing documentation...")
    text_content = process_page_content(html_content)
    
    print("Creating vector store...")
    create_vector_store(text_content)
    
    print("Documentation processing complete!")

if __name__ == "__main__":
    main()