# rkllm-just-works

A set of simple Python scripts to run RKLLM (Rockchip Large Language Model) with OpenAI-compatible API, verified on rk3588, RKNPU v0.9.6, and RKLLM v1.2.0.

![space](assets/space.png)


---

## Project Goal

A simple server to inference RKLLM models, with memory built-in in the server lifetime.

## Repository Structure

- `server.py` / `server_memory.py`  
  Main Flask server scripts exposing an OpenAI-compatible chat API powered by RKLLM. Handles model initialization, chat history, and streaming responses.

- `convo.py`  
  Low-level Python interface for RKLLM inference using ctypes. Useful for direct invocation and testing.

- `client.py`  
  Simple HTTP client to interact with the running RKLLM server via API calls.

- `lite.py`  
  Minimal example using the OpenAI Python SDK to connect to the local RKLLM server.

- `reference/`  
  Reference implementations and demos:
  - `flask_server.py`: Full-featured Flask server example with multi-user support and advanced callback handling.
  - `chat_api_flask.py`: Example Python client for sending chat requests to the server, including streaming and non-streaming modes.
  - `llm_demo.cpp`: C++ demo for direct inference with RKLLM.
  
- `assets/`  
  Images and other static resources (e.g., `space.png` for README illustration).

- `LICENSE`  
  Apache 2.0 license for usage and distribution.

---

## Usage

1. **Download an RKLLM model**  
   Obtain your `.bin` or other RKLLM-compatible model file.

2. **Place the model file**  
   Put your model (e.g., `Qwen3-0.6B-RKLLM.rkllm`) in a directory of your choice, for example:
   ```
   ./models/Qwen3-0.6B-RKLLM.rkllm
   ```

3. **Edit the server scripts to point to your model**  
   In `server.py` or `server_memory.py`, set the `model_path` parameter to the correct path, such as:
   ```python
   MODEL_PATH = "./models/Qwen3-0.6B-RKLLM.bin"
   ```

4. **Start the server**  
   Run the server:
   ```
   python3 server.py
   ```
   or (for persistent memory version)
   ```
   python3 server_memory.py
   ```

5. **Interact with the server**  
   - Use `client.py` or `lite.py` to send requests.
   - For OpenAI SDK compatibility, set your `base_url` to `http://0.0.0.0:1306/v1`.

6. **Example using the client:**
   ```
   python3 client.py
   ```

**Note:**  
- Make sure `librkllmrt.so` (the RKLLM runtime) is present in `./src/` or update the path in the scripts.
- If using custom models or multiple models, adjust the `model_path` and related parameters accordingly.
- [Model Collection](https://huggingface.co/collections/ThomasTheMaker/rkllm-v120-681974c057d4de18fb38be6c)

---

