"""
Conversation History Management

This module handles conversation history tracking for the RKLLM server.
"""

import threading
from typing import List, Dict, Any, Optional
import time
import uuid

class ConversationManager:
    """
    Manages conversation histories for different chat sessions
    """
    def __init__(self):
        self.conversations = {}  # Dict[conversation_id, conversation_history]
        self.lock = threading.RLock()
        
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a given ID, creating it if it doesn't exist
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            The conversation history as a list of message objects
        """
        with self.lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            return self.conversations[conversation_id]
    
    def add_message(self, 
                   conversation_id: str, 
                   role: str, 
                   content: str, 
                   tool_calls: Optional[List[Dict]] = None) -> None:
        """
        Add a message to the conversation history
        
        Args:
            conversation_id: The ID of the conversation
            role: The role of the message ('system', 'user', 'assistant', or 'tool')
            content: The content of the message
            tool_calls: Optional list of tool calls (only for assistant messages)
        """
        with self.lock:
            conversation = self.get_conversation(conversation_id)
            
            message = {"role": role, "content": content}
            
            # Add tool calls if specified (only for assistant messages)
            if tool_calls and role == "assistant":
                message["tool_calls"] = tool_calls
                
            conversation.append(message)
    
    def add_tool_result(self, 
                       conversation_id: str, 
                       tool_call_id: str, 
                       function_name: str, 
                       content: str) -> None:
        """
        Add a tool result to the conversation history
        
        Args:
            conversation_id: The ID of the conversation
            tool_call_id: The ID of the tool call this is responding to
            function_name: The name of the function called
            content: The content/result of the tool call
        """
        with self.lock:
            conversation = self.get_conversation(conversation_id)
            
            message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": content
            }
            
            conversation.append(message)
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear the conversation history for a given ID
        
        Args:
            conversation_id: The ID of the conversation to clear
        """
        with self.lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
    
    def generate_id(self) -> str:
        """
        Generate a unique conversation ID
        
        Returns:
            A new unique conversation ID
        """
        return str(uuid.uuid4())

# Global instance
conversation_manager = ConversationManager()
