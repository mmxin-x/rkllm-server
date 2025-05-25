import openai # openai v1.0.0+
import sys

# Import settings from config.py
from config import *

# Configure OpenAI client with local server URL
client = openai.OpenAI(api_key=API_KEY, base_url=f"http://{SERVER_HOST}:{SERVER_PORT}{API_BASE_PATH}")

# Set up message
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
]

print("Sending request with streaming enabled...") 
print("User: this is a test request, write a short poem")
print("Assistant: ", end="", flush=True)

# Try to fetch available models from the server
try:
    models_response = client.models.list()
    model_name = models_response.data[0].id if models_response.data else "Qwen3-0.6B-RKLLM"
except Exception:
    # Fallback to default model name if we can't get the list
    model_name = "Qwen3-0.6B-RKLLM"

print(f"Using model: {model_name}")

# Create streaming response
for chunk in client.chat.completions.create(
    model=model_name,
    messages=messages,
    stream=True
):
    # Extract and print the content from each chunk
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)
        
print("\n\nStreaming completed.")