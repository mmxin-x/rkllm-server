#!/usr/bin/env python3
"""
RKLLM Server - Main Entry Point

This script starts the RKLLM server with OpenAI-compatible API endpoints.
"""

from rkllm.app import run_server

if __name__ == "__main__":
    run_server()
