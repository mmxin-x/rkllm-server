#!/bin/bash
# Simple script to run the RKLLM server

# Ensure Python environment is activated if using virtualenv
# source ./myenv/bin/activate

# Optional: Set environment variables for configuration
# export RKLLM_SERVER_PORT=1306
# export RKLLM_MODEL_PATH="../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"

# Run the server
python3 simple_server.py
