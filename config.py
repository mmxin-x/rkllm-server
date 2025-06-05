"""
Configuration settings for the RKLLM server.
This file centralizes all configurable parameters to make them easier to manage.
"""

# Model Configuration
MODEL_PATH = "../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
LIBRARY_PATH = "./src/librkllmrt.so"  # Path to the RKLLM runtime library

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 1306
API_BASE_PATH = "/v1"
API_KEY = "anything"  # Default API key for authentication (can be any string)

# Model Parameters
MAX_CONTEXT_LENGTH = 30000
MAX_NEW_TOKENS = -1  # -1 means no limit
N_KEEP = 32
USE_GPU = True
IS_ASYNC = False

# System Prompt (if any)
SYSTEM_PROMPT = "You are a helpful assistant."

# Debug Configuration
DEBUG_MODE = False
LOG_LEVEL = 2  # 0: Error, 1: Warning, 2: Info, 3: Debug
