#!/usr/bin/env python3
import json
import requests
import time
import argparse
from typing import List, Dict, Any, Optional

# Base URL for the RKLLM server
BASE_URL = "http://localhost:1306/v1"

def test_models_endpoint():
    """Test the /models endpoint to verify tool calling capability is reported"""
    print("\n=== Testing Models Endpoint ===")
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code != 200:
        print(f"Error accessing models endpoint: {response.status_code}")
        print(response.text)
        return False
    
    models_data = response.json()
    print(json.dumps(models_data, indent=2))
    
    # Check if at least one model has tool_calling capability
    has_tool_calling = any(
        model.get("capabilities", {}).get("tool_calling", False) 
        for model in models_data.get("data", [])
    )
    
    print(f"Tool calling capability reported: {has_tool_calling}")
    return has_tool_calling

def test_non_streaming_tool_call():
    """Test non-streaming tool calling"""
    print("\n=== Testing Non-Streaming Tool Calling ===")
    
    # Define a weather tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Create the chat request
    request_data = {
        "model": "rkllm_model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in New York City?"}
        ],
        "tools": tools,
        "stream": False
    }
    
    print("Sending request...")
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/chat/completions", json=request_data)
    elapsed = time.time() - start_time
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False
    
    response_data = response.json()
    print(f"Response received in {elapsed:.2f} seconds")
    
    # Check if tool_calls is in the response
    has_tool_calls = "tool_calls" in response_data["choices"][0].get("message", {})
    print(f"Has tool calls: {has_tool_calls}")
    
    # Print the full response in a readable format
    print("\nResponse:")
    print(json.dumps(response_data, indent=2))
    
    return has_tool_calls

def test_streaming_tool_call():
    """Test streaming tool calling"""
    print("\n=== Testing Streaming Tool Calling ===")
    
    # Define a search tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search for documents based on a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Create the chat request
    request_data = {
        "model": "rkllm_model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for documents about climate change"}
        ],
        "tools": tools,
        "stream": True
    }
    
    print("Sending streaming request...")
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/chat/completions", json=request_data, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False
    
    print("Receiving stream events:")
    content_chunks = []
    tool_calls = []
    finish_reason = None
    
    try:
        for chunk in response.iter_lines():
            if chunk:
                # Filter out keep-alive new lines
                if chunk.startswith(b"data:"):
                    data_str = chunk[5:].decode('utf-8').strip()
                    if data_str != "[DONE]":
                        data = json.loads(data_str)
                        
                        # Check for content
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            content_chunks.append(delta["content"])
                            print(f"Content: {delta['content']}", end="", flush=True)
                        
                        # Check for tool calls
                        if "tool_calls" in delta:
                            tool_call = delta["tool_calls"][0]
                            
                            # Print tool call information
                            if "function" in tool_call:
                                function_info = tool_call["function"]
                                if "name" in function_info:
                                    print(f"\nTool call - Function: {function_info['name']}")
                                if "arguments" in function_info:
                                    args = json.loads(function_info['arguments'])
                                    print(f"Arguments: {json.dumps(args, indent=2)}")
                                    
                            tool_calls.append(tool_call)
                        
                        # Check for finish reason
                        if data["choices"][0]["finish_reason"] is not None:
                            finish_reason = data["choices"][0]["finish_reason"]
                            print(f"\nFinish reason: {finish_reason}")
    except Exception as e:
        print(f"Error parsing streaming response: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nStreaming completed in {elapsed:.2f} seconds")
    
    has_tool_calls = len(tool_calls) > 0
    print(f"Has tool calls: {has_tool_calls}")
    print(f"Finish reason was: {finish_reason}")
    
    return has_tool_calls and finish_reason == "tool_calls"

def test_tool_choice():
    """Test tool_choice parameter to require a specific tool"""
    print("\n=== Testing Tool Choice Parameter ===")
    
    # Define multiple tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Create the chat request with tool_choice
    request_data = {
        "model": "rkllm_model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in Paris?"}
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_current_weather"}},
        "stream": False
    }
    
    print("Sending request with tool_choice...")
    response = requests.post(f"{BASE_URL}/chat/completions", json=request_data)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False
    
    response_data = response.json()
    
    # Check if the right tool was called
    tool_calls = response_data["choices"][0].get("message", {}).get("tool_calls", [])
    correct_tool_used = False
    
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.get("function", {}).get("name") == "get_current_weather":
                correct_tool_used = True
                break
    
    print(f"Used correct tool (get_current_weather): {correct_tool_used}")
    
    # Print the response
    print("\nResponse:")
    print(json.dumps(response_data, indent=2))
    
    return correct_tool_used

def main():
    parser = argparse.ArgumentParser(description='Test tool calling functionality of RKLLM server')
    parser.add_argument('--url', default='http://localhost:1306/v1', help='Base URL for the RKLLM server')
    parser.add_argument('--test', choices=['all', 'models', 'non-streaming', 'streaming', 'tool-choice'], 
                        default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    print(f"Testing RKLLM server at {BASE_URL}")
    
    # Run the selected test(s)
    if args.test in ['all', 'models']:
        test_models_endpoint()
    
    if args.test in ['all', 'non-streaming']:
        test_non_streaming_tool_call()
    
    if args.test in ['all', 'streaming']:
        test_streaming_tool_call()
    
    if args.test in ['all', 'tool-choice']:
        test_tool_choice()

if __name__ == "__main__":
    main()
