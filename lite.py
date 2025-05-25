import openai # openai v1.0.0+
import sys

# Configure OpenAI client with local server URL
client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:1306/v1")

# Set up message
messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
]

print("Sending request with streaming enabled...") 
print("User: this is a test request, write a short poem")
print("Assistant: ", end="", flush=True)

# Create streaming response
for chunk in client.chat.completions.create(
    model="Qwen3-0.6B-RKLLM",
    messages=messages,
    stream=True
):
    # Extract and print the content from each chunk
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)
        
print("\n\nStreaming completed.")