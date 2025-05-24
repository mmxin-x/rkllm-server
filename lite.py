import litellm
import os
import sys

# --- Configuration for your local RKLLM OpenAI-compliant server ---

# LiteLLM expects the api_base to be the base URL of the OpenAI-compatible API.
# It will typically append standard paths like /chat/completions.
# For OpenAI-compatible endpoints, this is usually the URL up to /v1.
CUSTOM_API_BASE = "http://127.0.0.1:1306/v1" 

# The model name here is for litellm. Your server currently uses one globally loaded model
# and ignores the model name sent in the request.
# Using "openai/" prefix can help litellm identify it as an OpenAI-like API.
# You can use any descriptive name.
CUSTOM_MODEL_NAME = "openai/rkllm-qwen-custom" 

# Your server doesn't use an API key, but litellm might require one to be set
# for some configurations, or it might default to environment variables.
# Providing a dummy key is often a safe bet for custom OpenAI-like endpoints.
CUSTOM_API_KEY = "dummy_key_for_litellm"


def call_litellm_one_shot(user_message):
    """Calls the API for a non-streaming (one-shot) response using litellm."""
    messages = [{"role": "user", "content": user_message}]
    try:
        print(f"\n[LiteLLM Client] Sending one-shot request to: {CUSTOM_API_BASE}")
        print(f"[LiteLLM Client] Model for litellm: {CUSTOM_MODEL_NAME}")
        
        response = litellm.completion(
            model=CUSTOM_MODEL_NAME,
            messages=messages,
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY,
            # custom_llm_provider="openai", # Often not needed if model starts with "openai/" or api_base is set
            stream=False,
            timeout=120 # seconds, for the entire request
        )
        
        # litellm.completion returns a ModelResponse object for non-streaming
        # Access content via response.choices[0].message.content
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            assistant_message = response.choices[0].message.content
            print("\nAssistant (LiteLLM One-shot):")
            print(assistant_message)
            if response.usage:
                 print(f"\nUsage (from litellm): {response.usage}") # Usage is estimated by your server
        else:
            print("\n[LiteLLM Client] Received an empty or unexpected response structure:")
            print(f"Full response object: {response}")

    except litellm.exceptions.APIConnectionError as e:
        print(f"\n[LiteLLM Client Error] API Connection Error: {e}")
        print("Ensure your server is running at the specified CUSTOM_API_BASE.")
    except Exception as e:
        print(f"\n[LiteLLM Client Error] An unexpected error occurred: {e}")
        # For more detailed litellm logs for debugging, you can set:
        # litellm.set_verbose = True 
        # before making the call.

def call_litellm_streaming(user_message):
    """Calls the API for a streaming response using litellm."""
    messages = [{"role": "user", "content": user_message}]
    print("\nAssistant (LiteLLM Streaming):")
    try:
        print(f"\n[LiteLLM Client] Sending streaming request to: {CUSTOM_API_BASE}")
        print(f"[LiteLLM Client] Model for litellm: {CUSTOM_MODEL_NAME}")

        response_stream = litellm.completion(
            model=CUSTOM_MODEL_NAME,
            messages=messages,
            api_base=CUSTOM_API_BASE,
            api_key=CUSTOM_API_KEY,
            # custom_llm_provider="openai",
            stream=True,
            timeout=120 # seconds, for the entire request duration
        )
        
        # Iterate through the stream
        for chunk in response_stream:
            # Each chunk is a ModelResponse object (specifically a StreamingChunk)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                print(content_chunk, end="", flush=True)
            
            # Check for finish reason (optional, as [DONE] is also a signal)
            if chunk.choices and chunk.choices[0].finish_reason:
                # This might not always be present in the last content chunk itself,
                # but rather in a separate final chunk with empty delta.
                # The loop will naturally end when the stream sends [DONE] or closes.
                # print(f"\n<Stream received finish_reason: {chunk.choices[0].finish_reason}>") # For debugging
                pass
        
        print() # Ensure a newline after streaming is complete

    except litellm.exceptions.APIConnectionError as e:
        print(f"\n[LiteLLM Client Error] API Connection Error: {e}")
        print("Ensure your server is running at the specified CUSTOM_API_BASE.")
    except Exception as e:
        print(f"\n[LiteLLM Client Error] An unexpected error occurred during streaming: {e}")
        # litellm.set_verbose = True


if __name__ == "__main__":

    print("\nLiteLLM Client for Custom OpenAI-Compliant RKLLM Server")
    print(f"Targeting API Base: {CUSTOM_API_BASE}")
    print(f"Using Model Identifier for LiteLLM: {CUSTOM_MODEL_NAME}")
    
    # Alternative: Set global config for litellm if you prefer not to pass in each call
    # litellm.api_base = CUSTOM_API_BASE 
    # litellm.api_key = CUSTOM_API_KEY
    # litellm.add_model_list = [
    #    {
    #        "model_name": CUSTOM_MODEL_NAME, # The name you use in litellm.completion
    #        "litellm_params": {
    #            "model": CUSTOM_MODEL_NAME, # Actual model identifier for the API call if needed, often same
    #            "api_base": CUSTOM_API_BASE,
    #            "api_key": CUSTOM_API_KEY,
    #        }
    #    }
    # ]


    while True:
        try:
            user_input = input("\nEnter your message (or type 'quit' to exit): ")
            if user_input.lower() == 'quit':
                print("Exiting client.")
                break
            if not user_input.strip():
                continue

            stream_choice_input = input("Stream response? (yes/no) [yes]: ").lower().strip()
            should_stream = not (stream_choice_input == "n" or stream_choice_input == "no")


            if should_stream:
                call_litellm_streaming(user_input)
            else:
                call_litellm_one_shot(user_input)

        except KeyboardInterrupt:
            print("\nExiting client...")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the client's main loop: {e}")

