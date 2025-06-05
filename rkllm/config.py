"""
Configuration settings for the RKLLM server.
This file centralizes all configurable parameters to make them easier to manage.
"""
import os

# Model Configuration
MODEL_PATH = os.environ.get("RKLLM_MODEL_PATH", "../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm")
LIBRARY_PATH = os.environ.get("RKLLM_LIBRARY_PATH", "./src/librkllmrt.so")

# Server Configuration
SERVER_HOST = os.environ.get("RKLLM_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("RKLLM_SERVER_PORT", 1306))
API_BASE_PATH = os.environ.get("RKLLM_API_BASE_PATH", "/v1")
API_KEY = os.environ.get("RKLLM_API_KEY", "anything")  # Default API key for authentication

# Model Parameters
MAX_CONTEXT_LENGTH = int(os.environ.get("RKLLM_MAX_CONTEXT_LENGTH", 16000))
MAX_NEW_TOKENS = int(os.environ.get("RKLLM_MAX_NEW_TOKENS", -1))  # -1 means no limit
N_KEEP = int(os.environ.get("RKLLM_N_KEEP", 16600))
USE_GPU = os.environ.get("RKLLM_USE_GPU", "True").lower() == "true"
IS_ASYNC = os.environ.get("RKLLM_IS_ASYNC", "False").lower() == "true"

# System Prompt (if any)
SYSTEM_PROMPT = os.environ.get("RKLLM_SYSTEM_PROMPT", "You are a helpful assistant.")

# Debug Configuration
DEBUG_MODE = os.environ.get("RKLLM_DEBUG_MODE", "False").lower() == "true"
LOG_LEVEL = int(os.environ.get("RKLLM_LOG_LEVEL", 2))  # 0: Error, 1: Warning, 2: Info, 3: Debug
