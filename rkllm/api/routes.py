"""
API Routes for RKLLM Server

This module defines the Flask routes for the RKLLM OpenAI-compatible API.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Generator
import flask
from flask import request, jsonify, Response, stream_with_context

from ..config import SYSTEM_PROMPT as DEFAULT_SYSTEM_MESSAGE, MAX_NEW_TOKENS
from ..lib.rkllm_runtime import runtime
from ..utils.tool_calling import try_parse_tool_calls, format_tool_calls_for_response, create_tools_system_prompt
from ..utils.conversation import conversation_manager

# Constants
USER_TURN_START = "<|im_start|>user\n".encode('utf-8')
USER_TURN_END = "<|im_end|>".encode('utf-8')
ASSISTANT_TURN_START = "<|im_start|>assistant\n".encode('utf-8')
ASSISTANT_TURN_END = "<|im_end|>".encode('utf-8')

# Thread-local storage for current stream response
thread_local = threading.local()

# Store current tool calls for streaming
current_tool_calls = {}
current_tool_calls_lock = threading.Lock()

def setup_routes(app):
    @app.route('/v1/models', methods=['GET'])
    def models_handler():
        """
        Handle GET requests to /v1/models
        Returns information about available models
        """
        model_id = "Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
        
        return jsonify({
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rkllm",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                    "capabilities": {
                        "tool_calling": True
                    }
                }
            ]
        })

    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions_handler():
        """
        Handle POST requests to /v1/chat/completions
        Processes chat completion requests and returns responses
        """
        request_data = request.json
        
        # Get or generate a conversation ID
        conversation_id = request_data.get("conversation_id", conversation_manager.generate_id())
        
        # Extract messages from the request
        messages = request_data.get("messages", [])
        
        # Check for tools or functions parameters (functions is deprecated but supported for compatibility)
        tools = request_data.get("tools", None)
        functions = request_data.get("functions", None)
        tool_choice = request_data.get("tool_choice", None)
        
        # If no messages are provided, return an error
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg
                break
        
        if not last_user_message:
            return jsonify({"error": "No user message found"}), 400
        
        last_user_message_content = last_user_message.get("content", "")
        
        # Format the user message
        current_user_turn_bytes = USER_TURN_START + last_user_message_content.encode('utf-8') + USER_TURN_END

        # Initialize system message before any modifications
        system_message = DEFAULT_SYSTEM_MESSAGE

        # Custom system message in request
        if "system" in request_data and request_data["system"]:
            system_message = request_data["system"]

        # If tools are provided, add them to the system prompt
        if tools or functions:
            try:
                # Prefer tools over functions if both are provided (tools is the newer format)
                tools_to_use = tools if tools else []
                
                # If tools is not provided but functions is, convert functions to tools format
                if not tools and functions:
                    tools_to_use = [{"function": func, "type": "function"} for func in functions]
                
                # Create tools system prompt
                tools_info = create_tools_system_prompt(tools_to_use)
                
                # Append tools info to system message
                if tools_info:
                    system_message = f"{system_message}\n\n{tools_info}"
                    
                # If tool_choice is "required", add instruction to use a tool
                if tool_choice and tool_choice != "none" and tool_choice != "auto":
                    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                        func_name = tool_choice.get("function", {}).get("name", "")
                        if func_name:
                            system_message += f"\n\nYou MUST use the '{func_name}' tool for this request."
            except Exception as e:
                print(f"Error adding tools to system prompt: {e}")
                # Continue without tools if there's an error
        
        # Format the current prompt using the history and the new user message
        history_text = b''
        
        # Get conversation history
        conversation_history = conversation_manager.get_conversation(conversation_id)
        
        # Add messages from the request that aren't already in the conversation history
        # (Only if conversation history is empty - otherwise we assume it's already up to date)
        if not conversation_history:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Skip if role or content is empty
                if not role or not content:
                    continue
                    
                # Add system, user, and assistant messages
                if role in ["system", "user", "assistant"]:
                    tool_calls = msg.get("tool_calls", None)
                    conversation_manager.add_message(conversation_id, role, content, tool_calls)
                    
                # Add tool results
                elif role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    function_name = msg.get("name", "")
                    
                    if tool_call_id and function_name:
                        conversation_manager.add_tool_result(
                            conversation_id, tool_call_id, function_name, content
                        )
        
        # Check if streaming is requested
        stream = request_data.get("stream", False)
        
        # Format the history from conversation
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # System message is handled separately
                pass
            elif role == "user":
                history_text += USER_TURN_START + content.encode('utf-8') + USER_TURN_END
            elif role == "assistant":
                # Only include the actual content, not the tool calls
                history_text += ASSISTANT_TURN_START + content.encode('utf-8') + ASSISTANT_TURN_END
            elif role == "tool":
                # Tool results are included in the prompt
                tool_call_id = msg.get("tool_call_id", "")
                function_name = msg.get("name", "")
                tool_result = f"<tool_result>\n{content}\n</tool_result>"
                history_text += tool_result.encode('utf-8')
        
        # Build the final prompt
        system_message_bytes = f"<|im_start|>system\n{system_message}<|im_end|>\n".encode('utf-8')
        final_prompt = system_message_bytes + history_text + current_user_turn_bytes + ASSISTANT_TURN_START
        
        # Generation parameters
        temperature = float(request_data.get("temperature", 0.8))
        max_tokens = int(request_data.get("max_tokens", MAX_NEW_TOKENS))
        if max_tokens < 0:
            max_tokens = MAX_NEW_TOKENS
        
        # Process the request
        if stream:
            return process_streaming_request(final_prompt, conversation_id, max_tokens, temperature)
        else:
            return process_non_streaming_request(final_prompt, conversation_id, max_tokens, temperature)

    def process_non_streaming_request(final_prompt, conversation_id, max_tokens, temperature):
        """
        Process a non-streaming chat completion request
        """
        result_text = ""
        error_message = None
        
        def callback(result, userdata, status):
            nonlocal result_text, error_message
            
            if status == 1:  # RKLLM_RESULT_FINISH
                pass
            elif status == 2:  # RKLLM_RESULT_ERROR
                error_message = "LLM inference error"
            else:
                if result.contents.text:
                    result_text += result.contents.text.decode('utf-8', errors='replace')
        
        # Run inference
        result = runtime.run_inference(final_prompt.decode('utf-8'), callback, 0, True)
        
        if result != 0 or error_message:
            return jsonify({"error": error_message or "Unknown error"}), 500
        
        # Parse any tool calls in the response
        parsed = try_parse_tool_calls(result_text)
        content = parsed["content"]
        tool_calls = parsed["tool_calls"]
        
        # Update conversation history
        conversation_manager.add_message(conversation_id, "assistant", content, tool_calls if tool_calls else None)
        
        # Format the response
        response = {
            "id": f"chatcmpl-{str(int(time.time() * 1000000))[-16:]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm",
            "choices": [
                {
                    "index": 0,
                    "message": format_tool_calls_for_response(content, tool_calls),
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(final_prompt) // 4,  # Rough estimate
                "completion_tokens": len(result_text) // 4,  # Rough estimate
                "total_tokens": (len(final_prompt) + len(result_text)) // 4  # Rough estimate
            }
        }
        
        return jsonify(response)

    def process_streaming_request(final_prompt, conversation_id, max_tokens, temperature):
        """
        Process a streaming chat completion request
        """
        # Create response headers
        response_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        
        # Clear any existing tool calls for this thread
        with current_tool_calls_lock:
            thread_id = threading.get_ident()
            current_tool_calls[thread_id] = []
        
        # Create the streaming response
        return Response(
            stream_with_context(generate_streaming_response(final_prompt, conversation_id)),
            headers=response_headers
        )

    def generate_streaming_response(prompt, conversation_id):
        """
        Generator for streaming responses
        """
        result_text = ""
        is_finished = False
        error_occurred = False
        
        thread_id = threading.get_ident()
        thread_local.result_text = ""
        
        def callback(result, userdata, status):
            if status == 1:  # RKLLM_RESULT_FINISH
                nonlocal is_finished
                is_finished = True
            elif status == 2:  # RKLLM_RESULT_ERROR
                nonlocal error_occurred
                error_occurred = True
            else:
                if result.contents.text:
                    text = result.contents.text.decode('utf-8', errors='replace')
                    thread_local.result_text += text
                    
                    # Try to parse tool calls in the accumulated text
                    with current_tool_calls_lock:
                        parsed = try_parse_tool_calls(thread_local.result_text)
                        current_tool_calls[thread_id] = parsed["tool_calls"]
        
        # Run inference
        result = runtime.run_inference(prompt.decode('utf-8'), callback, 0, True)
        
        if result != 0:
            yield f"data: {json.dumps({'error': 'Failed to start inference'})}\n\n"
            return
        
        # Initial response
        response_start = {
            "id": f"chatcmpl-{str(int(time.time() * 1000000))[-16:]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(response_start)}\n\n"
        
        # Stream the response
        last_sent_length = 0
        chunk_size = 8  # Number of characters to send per chunk
        has_sent_content = False
        
        # Check for tool calls periodically
        tool_calls_sent = False
        
        while not is_finished and not error_occurred:
            current_length = len(thread_local.result_text)
            
            if current_length > last_sent_length:
                # We have new text to send
                chunk = thread_local.result_text[last_sent_length:current_length]
                last_sent_length = current_length
                
                # Check for tool calls
                with current_tool_calls_lock:
                    tool_calls_list = current_tool_calls.get(thread_id, [])
                
                # If we have tool calls and haven't sent them yet
                if tool_calls_list and not tool_calls_sent:
                    # Check if we've accumulated most of the response
                    # (This is a heuristic - we assume tool calls come at the end)
                    if is_finished or len(chunk) < chunk_size:
                        # Parse the complete response to get content without tool calls
                        parsed = try_parse_tool_calls(thread_local.result_text)
                        content = parsed["content"]
                        
                        # Send content delta if we haven't sent all content yet
                        if content and not has_sent_content:
                            content_delta = {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(content_delta)}\n\n"
                            has_sent_content = True
                        
                        # Send tool calls delta
                        tool_calls_delta = {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": tool_calls_list},
                                    "finish_reason": "tool_calls"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(tool_calls_delta)}\n\n"
                        tool_calls_sent = True
                else:
                    # Regular content delta
                    if not has_sent_content:
                        content_delta = {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(content_delta)}\n\n"
            
            # Small delay to avoid CPU spinning
            time.sleep(0.01)
        
        # Update conversation history after streaming
        with current_tool_calls_lock:
            tool_calls_list = current_tool_calls.get(thread_id, [])
            
            # Parse the complete response one more time to get final content
            parsed = try_parse_tool_calls(thread_local.result_text)
            content = parsed["content"]
            
            # Update conversation history with the final content and tool calls
            conversation_manager.add_message(
                conversation_id, 
                "assistant", 
                content, 
                tool_calls_list if tool_calls_list else None
            )
            
            # Clean up
            if thread_id in current_tool_calls:
                del current_tool_calls[thread_id]
        
        # Send final done message
        if not tool_calls_sent:
            finish_chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"

    return app
