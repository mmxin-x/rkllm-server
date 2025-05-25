import requests
import json
import sys

# Import settings from config.py
from config import *

# Direct HTTP client configuration
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}{API_BASE_PATH}"

def test_conversation(use_streaming=False):
    try:
        print("Starting test conversation with the model...\n")
        
        # First, check that we can access the models endpoint
        try:
            response = requests.get(f"{BASE_URL}/models")
            if response.status_code == 200:
                print(f"Models available: {json.dumps(response.json(), indent=2)}\n")
            else:
                print(f"HTTP Error: {response.status_code} - {response.text}\n")
        except Exception as e:
            print(f"Error accessing models: {str(e)}\n")
        
        # Get user input for the conversation
        user_input = input("Enter your message (or press Enter to use default): ")
        if not user_input:
            user_input = "Tell me a short joke about programming."
        
        print(f"\nUser: {user_input}")
        print("Assistant: ", end="", flush=True)
        
        # Prepare the request payload
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": user_input}
        ]
        
        # Get model name from server response if available
        model_name = "Qwen3-0.6B-RKLLM"  # Default fallback model name
        try:
            # Use the model name from the server response if available
            if response.status_code == 200 and response.json().get("data") and len(response.json()["data"]) > 0:
                model_name = response.json()["data"][0]["id"]
        except Exception:
            pass
            
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": use_streaming
        }
        
        # Make the request to the chat completions endpoint
        if use_streaming:
            # Streaming mode
            print("\nUsing streaming mode...\n")
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            if response.status_code == 200:
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                                
                            try:
                                chunk = json.loads(data)
                                if chunk.get('choices') and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if delta and 'content' in delta and delta['content']:
                                        print(delta['content'], end='', flush=True)
                            except json.JSONDecodeError as e:
                                print(f"\nError decoding JSON: {e}\nRaw data: {data}")
            else:
                print(f"\nHTTP Error: {response.status_code} - {response.text}")
        else:
            # Non-streaming mode
            print("\nUsing non-streaming mode...\n")
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response headers: {response.headers}")
                print(f"Raw response: {json.dumps(result, indent=2)}")
                
                if result.get('choices') and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content', '')
                    if content and content.strip():
                        print(f"\nModel response: {content}")
                    else:
                        print("\n[No content received from model]")
                else:
                    print("\nNo choices found in response")
            else:
                print(f"\nHTTP Error: {response.status_code} - {response.text}")     
        print("\n\nConversation completed.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

# Run the test
if __name__ == "__main__":
    # Set to False to use non-streaming mode (more reliable for testing)
    test_conversation(use_streaming=True)