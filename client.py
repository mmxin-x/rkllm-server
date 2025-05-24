import requests
import json

API_URL = "http://127.0.0.1:1306/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def call_api_one_shot(user_message):
    """Calls the API for a non-streaming (one-shot) response."""
    payload = {
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "stream": False
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120) # 120 second timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        if data.get("choices") and len(data["choices"]) > 0:
            assistant_message = data["choices"][0].get("message", {}).get("content")
            print("\nAssistant (One-shot):")
            print(assistant_message)
            if data.get("usage"):
                print(f"\nUsage (estimated): {data['usage']}")
        else:
            print("\nReceived an empty or unexpected response:")
            print(data)

    except requests.exceptions.RequestException as e:
        print(f"\nError calling API: {e}")
    except json.JSONDecodeError:
        print("\nError: Could not decode JSON response from server.")
        print(f"Raw response: {response.text}")


def call_api_streaming(user_message):
    """Calls the API for a streaming response."""
    payload = {
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "stream": True
    }
    print("\nAssistant (Streaming):")
    try:
        with requests.post(API_URL, headers=HEADERS, json=payload, stream=True, timeout=120) as response: # 120 second timeout
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_data_str = decoded_line[len("data: "):]
                        if json_data_str.strip() == "[DONE]":
                            print("\n<End of stream>")
                            break
                        try:
                            json_data = json.loads(json_data_str)
                            if json_data.get("choices") and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})
                                content_chunk = delta.get("content")
                                if content_chunk:
                                    print(content_chunk, end="", flush=True)
                                if json_data["choices"][0].get("finish_reason"):
                                    print(f"\n<Stream finished with reason: {json_data['choices'][0]['finish_reason']}>")
                                    break # Exit loop once finish_reason is received
                        except json.JSONDecodeError:
                            # print(f"\nError decoding JSON chunk: {json_data_str}") # Can be noisy
                            pass # Silently ignore lines that are not valid JSON data chunks

    except requests.exceptions.RequestException as e:
        print(f"\nError calling API: {e}")
    except Exception as e_stream:
        print(f"\nAn unexpected error occurred during streaming: {e_stream}")
    finally:
        print() # Ensure a newline after streaming is complete or if an error occurs


if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nEnter your message (or type 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break

            stream_choice = input("Stream response? (yes/no) [yes]: ").lower().strip()
            if stream_choice == "" or stream_choice == "y" or stream_choice == "yes":
                call_api_streaming(user_input)
            elif stream_choice == "n" or stream_choice == "no":
                call_api_one_shot(user_input)
            else:
                print("Invalid choice for streaming. Please enter 'yes' or 'no'.")

        except KeyboardInterrupt:
            print("\nExiting client...")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the client: {e}")
