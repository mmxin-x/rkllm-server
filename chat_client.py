#!/usr/bin/env python3
"""
Interactive Chat Client for RKLLM Server

This client maintains conversation history across multiple turns
and provides a clean command-line interface for chatting with
the RKLLM model through its OpenAI-compatible API.
"""

import json
import requests
import argparse
import os
import sys
import time
from typing import Dict, List, Any, Optional

# API Constants
BASE_URL = "http://localhost:1306/v1"
DEFAULT_MODEL = "Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"

class ChatClient:
    def __init__(self, base_url: str = BASE_URL, model: str = DEFAULT_MODEL, streaming: bool = True):
        """Initialize the chat client"""
        self.base_url = base_url
        self.model = model
        self.streaming = streaming
        self.conversation_history = []
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        
    def add_system_message(self, content: str):
        """Set the system message for the conversation"""
        self.system_message = {"role": "system", "content": content}
        print(f"System message set to: {content}")
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        print("Conversation history has been reset.")
        
    def send_message(self, user_message: str) -> str:
        """Send a message to the model and get a response"""
        # Add the user message to the conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Build the full message history including system message
        messages = [self.system_message] + self.conversation_history
        
        # Create the request
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": self.streaming
        }
        
        try:
            full_response = ""
            
            # For streaming mode
            if self.streaming:
                response = requests.post(
                    f"{self.base_url}/chat/completions", 
                    json=request_data,
                    stream=True
                )
                
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    return f"Error: {response.status_code}"
                
                print("Assistant: ", end="", flush=True)
                for chunk in response.iter_lines():
                    if chunk:
                        # Filter out keep-alive new lines
                        if chunk.startswith(b"data:"):
                            data_str = chunk[5:].decode('utf-8').strip()
                            if data_str != "[DONE]":
                                data = json.loads(data_str)
                                delta = data["choices"][0]["delta"]
                                if "content" in delta:
                                    content = delta["content"]
                                    print(content, end="", flush=True)
                                    full_response += content
                                
                print()  # New line after response
            
            # For non-streaming mode
            else:
                response = requests.post(
                    f"{self.base_url}/chat/completions", 
                    json=request_data
                )
                
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    return f"Error: {response.status_code}"
                
                result = response.json()
                full_response = result["choices"][0]["message"]["content"]
                print(f"Assistant: {full_response}")
                
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": full_response})
            return full_response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n===== RKLLM Interactive Chat =====")
        print("Type your messages to chat with the model.")
        print("Special commands:")
        print("  /reset   - Reset the conversation")
        print("  /system <message> - Set a new system message")
        print("  /stream  - Toggle streaming mode")
        print("  /exit    - Exit the chat")
        print("=====================================\n")
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                # Handle special commands
                if user_input.strip() == "/exit":
                    print("Exiting chat. Goodbye!")
                    break
                    
                elif user_input.strip() == "/reset":
                    self.reset_conversation()
                    continue
                    
                elif user_input.strip() == "/stream":
                    self.streaming = not self.streaming
                    print(f"Streaming mode: {'ON' if self.streaming else 'OFF'}")
                    continue
                    
                elif user_input.startswith("/system "):
                    system_content = user_input[8:].strip()
                    self.add_system_message(system_content)
                    continue
                
                # Regular message - send to model
                if user_input.strip():
                    self.send_message(user_input)
            
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting chat.")
                break
            
            except Exception as e:
                print(f"\nError: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Interactive Chat Client for RKLLM Server")
    parser.add_argument("--url", default=BASE_URL, help=f"Base URL for the API (default: {BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")
    parser.add_argument("--system", default=None, help="Initial system message")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create client
    client = ChatClient(
        base_url=args.url,
        model=args.model,
        streaming=not args.no_stream
    )
    
    # Set custom system message if provided
    if args.system:
        client.add_system_message(args.system)
    
    # Start interactive chat session
    client.interactive_chat()
