#!/usr/bin/env python3
"""
Example OpenAI client for the RKLLM server.

This example demonstrates how to use the OpenAI Python SDK to connect
to the local RKLLM server running on port 1306.
"""

import os
from openai import OpenAI

# Create a client instance with the local server URL
client = OpenAI(
    api_key="anything",  # The server accepts any API key
    base_url="http://localhost:1306/v1"
)

def chat_completion(prompt, system_message=None):
    """
    Send a chat completion request to the RKLLM server.
    
    Args:
        prompt: The user message to send
        system_message: Optional system message
        
    Returns:
        The model's response
    """
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add user message
    messages.append({"role": "user", "content": prompt})
    
    # Send the request
    response = client.chat.completions.create(
        model="Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm",  # Model name (will be ignored by server)
        messages=messages
    )
    
    return response.choices[0].message.content

def chat_completion_with_tool(prompt, tool_definition):
    """
    Send a chat completion request with tools to the RKLLM server.
    
    Args:
        prompt: The user message to send
        tool_definition: Tool definition following OpenAI format
        
    Returns:
        The response, including any tool calls
    """
    # Define messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": prompt}
    ]
    
    # Send the request
    response = client.chat.completions.create(
        model="Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm",
        messages=messages,
        tools=[tool_definition]
    )
    
    return response

def streaming_chat(prompt, system_message=None):
    """
    Send a streaming chat completion request to the RKLLM server.
    
    Args:
        prompt: The user message to send
        system_message: Optional system message
    """
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add user message
    messages.append({"role": "user", "content": prompt})
    
    # Send the streaming request
    stream = client.chat.completions.create(
        model="Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm",
        messages=messages,
        stream=True
    )
    
    print("Streaming response:")
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    # Example 1: Simple chat completion
    print("\n=== Example 1: Simple Chat ===")
    response = chat_completion(
        prompt="What is artificial intelligence?",
        system_message="You are a helpful, concise assistant."
    )
    print(f"Response: {response}\n")
    
    # Example 2: Chat with tool calling
    print("\n=== Example 2: Tool Calling ===")
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    tool_response = chat_completion_with_tool("Calculate 123 * 456", calculator_tool)
    
    # Check if we got a tool call
    tool_calls = tool_response.choices[0].message.tool_calls
    if tool_calls:
        print(f"Tool called: {tool_calls[0].function.name}")
        print(f"Arguments: {tool_calls[0].function.arguments}")
    else:
        print("No tool calls in the response")
        print(f"Content: {tool_response.choices[0].message.content}")
    
    # Example 3: Streaming chat
    print("\n=== Example 3: Streaming Chat ===")
    streaming_chat("Explain quantum computing in simple terms.")
