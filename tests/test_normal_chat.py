#!/usr/bin/env python3
# Simple test to verify normal chat functionality without tools

import requests
import json

BASE_URL = "http://localhost:1306/v1"

def test_normal_chat():
    print("=== Testing Normal Chat (No Tools) ===")
    print("Sending a simple chat request...")

    # Create a simple request with no tools
    request_data = {
        "model": "gpt-3.5-turbo",  # Will be ignored by the server, just for compatibility
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "stream": False
    }

    # Send the request to the server
    response = requests.post(f"{BASE_URL}/chat/completions", json=request_data)
    
    # Check the response
    if response.status_code == 200:
        response_json = response.json()
        print("Response status: OK")
        
        # Extract content
        content = response_json["choices"][0]["message"]["content"]
        finish_reason = response_json["choices"][0]["finish_reason"]
        
        # Check if there are any tool calls (shouldn't be)
        has_tool_calls = "tool_calls" in response_json["choices"][0]["message"]
        
        print(f"Has tool calls: {has_tool_calls}")
        print(f"Finish reason: {finish_reason}")
        print(f"Content: {content[:100]}...")  # Print first 100 chars
        
        print("\nFull Response:")
        print(json.dumps(response_json, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_normal_chat()
