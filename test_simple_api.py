#!/usr/bin/env python3
"""
Simple test script to verify the RKLLM server API is working
"""

import requests
import json

BASE_URL = "http://localhost:1306/v1"

def test_models_endpoint():
    """Test the models endpoint"""
    print("\n=== Testing Models Endpoint ===")
    response = requests.get(f"{BASE_URL}/models")
    
    if response.status_code == 200:
        print(f"Success! Status Code: {response.status_code}")
        print("Available Models:")
        models = response.json()["data"]
        for model in models:
            print(f"  - {model['id']}")
    else:
        print(f"Error! Status Code: {response.status_code}")
        print(response.text)

def test_chat_completion():
    """Test a basic chat completion request"""
    print("\n=== Testing Chat Completion ===")
    
    # Create a simple request
    request_data = {
        "model": "gpt-3.5-turbo",  # Model name is ignored by server
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, tell me a short joke."}
        ],
        "max_tokens": 100,
        "temperature": 0.8
    }
    
    # Send the request
    response = requests.post(
        f"{BASE_URL}/chat/completions", 
        json=request_data
    )
    
    if response.status_code == 200:
        print(f"Success! Status Code: {response.status_code}")
        
        resp_json = response.json()
        print("\nResponse Content:")
        print(resp_json["choices"][0]["message"]["content"])
        
        print("\nFull Response:")
        print(json.dumps(resp_json, indent=2))
    else:
        print(f"Error! Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Testing RKLLM Server API...")
    test_models_endpoint()
    test_chat_completion()
