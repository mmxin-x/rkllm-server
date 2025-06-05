#!/usr/bin/env python3
import json
import requests
import sys

# Base URL for the RKLLM server
BASE_URL = "http://localhost:1306/v1"

def test_simple_tool_call():
    """Simple direct test of tool calling with clear instructions"""
    print("=== Testing Simple Tool Call ===")
    
    # Define a calculator tool
    tools = [
        {
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
    ]
    
    # Create the chat request with explicit instruction to use the tool
    request_data = {
        "model": "rkllm_model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. When asked to calculate something, ALWAYS use the calculate function."},
            {"role": "user", "content": "Calculate 123 + 456"}
        ],
        "tools": tools,
        "temperature": 0.1,  # Lower temperature for more deterministic outputs
        "stream": False
    }
    
    print("Sending request to calculate 123 + 456...")
    response = requests.post(f"{BASE_URL}/chat/completions", json=request_data)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False
    
    response_data = response.json()
    
    # Check if tool_calls is in the response
    tool_calls = response_data["choices"][0].get("message", {}).get("tool_calls", [])
    has_tool_calls = len(tool_calls) > 0
    
    print(f"Has tool calls: {has_tool_calls}")
    
    if has_tool_calls:
        # Check if the function name is correct
        function_name = tool_calls[0].get("function", {}).get("name", "")
        print(f"Function called: {function_name}")
        
        # Check the arguments
        arguments = tool_calls[0].get("function", {}).get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except:
                pass
        
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        # Check finish reason
        finish_reason = response_data["choices"][0].get("finish_reason")
        print(f"Finish reason: {finish_reason}")
    
    # Print the full response
    print("\nFull Response:")
    print(json.dumps(response_data, indent=2))
    
    return has_tool_calls

if __name__ == "__main__":
    test_simple_tool_call()
