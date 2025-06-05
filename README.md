# RKLLM Server

An OpenAI-compatible API server for running RKLLM (Rockchip Large Language Model) models on RK3588 platform.

![space](assets/space.png)

---

## Project Goal

A modular and well-structured server to run inference on RKLLM models, with proper support for:
- OpenAI-compatible API endpoints
- Tool/Function calling
- Streaming and non-streaming responses
- Conversation history management
- WebUI integration

## Repository Structure

This project follows a modern Python package structure:

- `rkllm/` - Main package directory
  - `__init__.py` - Package initialization
  - `app.py` - Flask application setup and server initialization
  - `config.py` - Centralized configuration module
  - `api/` - API endpoint implementations
    - `routes.py` - OpenAI-compatible API routes
  - `lib/` - Library interface code
    - `rkllm_runtime.py` - Low-level interface to RKLLM native library
  - `utils/` - Utility functions and helpers
    - `tool_calling.py` - Tool/function calling utilities
    - `conversation.py` - Conversation history management

- `main.py` - Main entry point to start the server

- `tests/` - Test scripts
  - `simple_tool_test.py` - Basic tool calling test
  - `test_tool_calling.py` - Comprehensive tool calling tests
  - `test_normal_chat.py` - Regular chat functionality tests

- `client.py` - Client script for interacting with the server
- `reference/` - Reference materials and documentation
- `assets/` - Images and other static resources
- `src/` - Native libraries folder containing `librkllmrt.so`

---

## Usage

1. **Download an RKLLM model**  
   Obtain your `.rkllm` model file.

2. **Place the model file**  
   Put your model (e.g., `Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm`) in a directory, for example:
   ```
   ../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm
   ```

3. **Set up environment variables (optional)**  
   You can configure the server using environment variables. For example:
   ```bash
   export RKLLM_MODEL_PATH="../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
   export RKLLM_SERVER_PORT=1306
   ```
   Alternatively, edit the configuration in `rkllm/config.py`.

4. **Start the server**  
   Run the server using the new entry point:
   ```bash
   python3 main.py
   ```

5. **Test the server**  
   Run one of the test scripts to verify functionality:
   ```bash
   # Test basic tool calling
   python3 tests/simple_tool_test.py
   
   # Test comprehensive tool calling features
   python3 tests/test_tool_calling.py
   
   # Test regular chat functionality
   python3 tests/test_normal_chat.py
   ```

6. **Use the API**  
   - The server exposes OpenAI-compatible endpoints at `http://0.0.0.0:1306/v1`
   - Use with any OpenAI client by setting the `base_url` to this address
   - Key endpoints:
     - `POST /v1/chat/completions` - Chat completions API
     - `GET /v1/models` - List available models
     
**Implementation Notes:**
- The simplified server (`simple_server.py`) provides the stability of the original implementation while fitting into the new modular structure
- Tool calling and function execution are fully supported in the simplified bridge implementation
- The fully modularized version is still under development

**Notes:**  
- Make sure `librkllmrt.so` (the RKLLM runtime) is present in `./src/` or update the library path in the configuration.
- For custom models, adjust the model path in the configuration.
- The server supports both streaming and non-streaming modes.
- Tool/function calling is fully supported with the Qwen3 model.
- [Model Collection](https://huggingface.co/collections/ThomasTheMaker/rkllm-v120-681974c057d4de18fb38be6c)

## Advanced Features

### Tool Calling
The server supports OpenAI-compatible function/tool calling with Qwen3 models. To use tool calling:

```python
import requests

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

response = requests.post(
    "http://localhost:1306/v1/chat/completions",
    json={
        "model": "gpt-3.5-turbo",  # Model name is ignored by the server
        "messages": [
            {"role": "user", "content": "Calculate 123 + 456"}
        ],
        "tools": tools,
        "stream": False
    }
)

print(response.json())
```

---

