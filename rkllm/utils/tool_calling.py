"""
Tool Calling Utilities

This module provides functions for parsing and formatting tool calls in RKLLM responses,
specifically for models like Qwen3 that support function calling.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union

def try_parse_tool_calls(content: str) -> Dict[str, Any]:
    """
    Parse tool calls from model output.
    
    Extracts tool calls enclosed in <tool_call> tags from the model's text output
    and formats them according to the OpenAI API standard.
    
    Args:
        content: The model's output text
    
    Returns:
        Dict containing:
            - content: Text content excluding tool calls
            - tool_calls: List of parsed tool calls (if any)
    """
    tool_calls = []
    content_parts = []
    last_end = 0
    
    try:
        # Find all tool calls using regex
        for i, m in enumerate(re.finditer(r"<tool_call>\n(.+?)?\n</tool_call>", content, re.DOTALL)):
            start, end = m.span()
            
            # Add content before this tool call to content_parts
            if start > last_end:
                content_parts.append(content[last_end:start])
            
            # Try to parse the tool call
            try:
                tool_json = m.group(1)
                func = json.loads(tool_json)
                
                # Ensure function name exists
                if "name" not in func:
                    print(f"Warning: Tool call missing 'name' field: {tool_json}")
                    continue
                    
                # Convert arguments from string to object if necessary
                if isinstance(func.get("arguments", ""), str) and func["arguments"].strip():
                    try:
                        func["arguments"] = json.loads(func["arguments"])
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse tool call arguments: {func['arguments']}")
                        # Keep arguments as string if parsing fails
                
                # Format tool call according to OpenAI API format
                tool_calls.append({
                    "id": f"call_{id(func):08x}",
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "arguments": json.dumps(func["arguments"]) if isinstance(func.get("arguments"), dict) else 
                                    func.get("arguments", "{}"),
                    }
                })
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                # Skip this tool call but continue processing others
            
            last_end = end
    
        # Add any remaining content after the last tool call
        if last_end < len(content):
            content_parts.append(content[last_end:])
        
        content_before_tools = "".join(content_parts).strip()
        
        return {
            "content": content_before_tools,
            "tool_calls": tool_calls
        }
    except Exception as e:
        print(f"Error in try_parse_tool_calls: {e}")
        # Return original content if parsing fails
        return {"content": content, "tool_calls": []}

def format_tool_calls_for_response(content: str, tool_calls: List[Dict]) -> Dict:
    """
    Format tool calls for the API response.
    
    Args:
        content: The content without tool calls
        tool_calls: List of parsed tool calls
    
    Returns:
        Dict formatted for API response with appropriate finish_reason
    """
    if tool_calls:
        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls"
        }
    else:
        return {
            "content": content,
            "finish_reason": "stop"
        }

def create_tools_system_prompt(tools: List[Dict]) -> str:
    """
    Create a system prompt extension for tools based on the provided tools list.
    
    Args:
        tools: List of tool definitions following OpenAI format
    
    Returns:
        String to append to system prompt with tool definitions
    """
    if not tools:
        return ""
    
    tools_info = "You have access to the following tools:\n\n"
    
    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "unknown")
        tool_description = tool.get("function", {}).get("description", "No description provided.")
        tool_params = tool.get("function", {}).get("parameters", {})
        
        tools_info += f"Tool: {tool_name}\n"
        tools_info += f"Description: {tool_description}\n"
        
        if "properties" in tool_params:
            tools_info += "Parameters:\n"
            for param_name, param_details in tool_params.get("properties", {}).items():
                param_description = param_details.get("description", "No description")
                param_type = param_details.get("type", "any")
                required = "required" if param_name in tool_params.get("required", []) else "optional"
                tools_info += f"  - {param_name}: ({param_type}, {required}) {param_description}\n"
        
        tools_info += "\n"
    
    # Add instructions on how to use tools
    tools_info += "\nWhen using a tool, use the exact format:\n"
    tools_info += "<tool_call>\n"
    tools_info += "{\"name\": \"function_name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}\n"
    tools_info += "</tool_call>\n"
    
    return tools_info
