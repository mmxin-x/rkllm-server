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

