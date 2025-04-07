"""
Memory utilities for AI Coach application
"""
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.conversation_storage import load_conversation, save_conversation

class ConversationBufferWindowMemory:
    """
    A memory class that keeps a sliding window of the conversation history.
    
    This class maintains the last k interactions (where an interaction is a user message
    followed by an assistant message) to provide recent conversation context.
    
    Attributes:
        k (int): The number of interactions to keep in memory
        conversation_id (str): The ID of the current conversation
        memory_key (str): The key used to identify the memory in the context
        return_messages (bool): Whether to return the memory as a list of messages or as a string
    """
    
    def __init__(
        self, 
        k: int = 5, 
        conversation_id: Optional[str] = None,
        memory_key: str = "history",
        return_messages: bool = False
    ):
        """
        Initialize the ConversationBufferWindowMemory.
        
        Args:
            k (int): The number of interactions to keep in memory (default: 5)
            conversation_id (str, optional): The ID of the conversation to load
            memory_key (str): The key used to identify the memory in the context
            return_messages (bool): Whether to return the memory as a list of messages or as a string
        """
        self.k = k
        self.conversation_id = conversation_id
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.buffer = []
        
        # Load conversation history if conversation_id is provided
        if conversation_id:
            self.load_memory_from_conversation(conversation_id)
    
    def load_memory_from_conversation(self, conversation_id: str) -> None:
        """
        Load memory from a saved conversation.
        
        Args:
            conversation_id (str): The ID of the conversation to load
        """
        self.conversation_id = conversation_id
        history = load_conversation(conversation_id)
        
        if history:
            # Convert the flat message list into interaction pairs
            interactions = []
            current_interaction = []
            
            for message in history:
                current_interaction.append(message)
                
                # When we have a user-assistant pair, add it to interactions
                if len(current_interaction) == 2 and current_interaction[0]['role'] == 'user' and current_interaction[1]['role'] == 'assistant':
                    interactions.append(current_interaction)
                    current_interaction = []
                
                # If we have an odd number of messages or the roles don't match,
                # start a new interaction with the current message
                if len(current_interaction) == 2 and (current_interaction[0]['role'] != 'user' or current_interaction[1]['role'] != 'assistant'):
                    current_interaction = [current_interaction[1]]
            
            # Keep only the last k interactions
            self.buffer = interactions[-self.k:] if len(interactions) > self.k else interactions
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        Add a message to the memory buffer.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The content of the message
            **kwargs: Additional message attributes
        """
        # Create message object
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        }
        
        # Add message to the appropriate interaction
        if role == 'user':
            # Start a new interaction with a user message
            self.buffer.append([message])
        elif role == 'assistant' and self.buffer and len(self.buffer[-1]) == 1 and self.buffer[-1][0]['role'] == 'user':
            # Complete the current interaction with an assistant message
            self.buffer[-1].append(message)
        else:
            # Handle edge cases (e.g., consecutive assistant messages)
            self.buffer.append([message])
        
        # Maintain the window size
        if len(self.buffer) > self.k:
            self.buffer = self.buffer[-self.k:]
        
        # Save to disk if conversation_id is set
        if self.conversation_id:
            self._save_to_disk()
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the memory buffer.
        
        Args:
            content (str): The content of the message
        """
        self.add_message('user', content)
    
    def add_ai_message(self, content: str, source: str = "Base LLM") -> None:
        """
        Add an AI message to the memory buffer.
        
        Args:
            content (str): The content of the message
            source (str): The source of the information (default: "Base LLM")
        """
        self.add_message('assistant', content, source=source)
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get the memory variables to include in the context.
        
        Returns:
            dict: The memory variables
        """
        if not self.buffer:
            return {self.memory_key: "" if not self.return_messages else []}
        
        if self.return_messages:
            # Return as a list of message objects
            messages = []
            for interaction in self.buffer:
                messages.extend(interaction)
            return {self.memory_key: messages}
        else:
            # Return as a formatted string
            result = []
            for interaction in self.buffer:
                if len(interaction) >= 1:
                    result.append(f"Human: {interaction[0]['content']}")
                if len(interaction) >= 2:
                    result.append(f"AI: {interaction[1]['content']}")
            return {self.memory_key: "\n".join(result)}
    
    def clear(self) -> None:
        """Clear the memory buffer."""
        self.buffer = []
    
    def _save_to_disk(self) -> None:
        """Save the current memory buffer to disk."""
        if not self.conversation_id:
            return
        
        # Convert buffer to flat message list
        messages = []
        for interaction in self.buffer:
            messages.extend(interaction)
        
        # Save to disk
        save_conversation(self.conversation_id, messages)
    
    def __str__(self) -> str:
        """String representation of the memory."""
        if not self.buffer:
            return "ConversationBufferWindowMemory(empty)"
        
        memory_str = self.get_memory_variables()[self.memory_key]
        if isinstance(memory_str, list):
            memory_str = "\n".join([f"{m['role']}: {m['content']}" for m in memory_str])
        
        return f"ConversationBufferWindowMemory(k={self.k}):\n{memory_str}"


def get_conversation_window(conversation_id: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Get the last k interactions from a conversation.
    
    Args:
        conversation_id (str): The ID of the conversation
        k (int): The number of interactions to retrieve
        
    Returns:
        list: The last k interactions
    """
    memory = ConversationBufferWindowMemory(k=k, conversation_id=conversation_id, return_messages=True)
    return memory.get_memory_variables()["history"]


def format_conversation_history(history: List[Dict[str, Any]]) -> str:
    """
    Format a conversation history as a string.
    
    Args:
        history (list): The conversation history
        
    Returns:
        str: The formatted conversation history
    """
    # If history is already a string, return it
    if isinstance(history, str):
        return history
        
    result = []
    for message in history:
        if message['role'] == 'user':
            result.append(f"Human: {message['content']}")
        else:
            result.append(f"AI: {message['content']}")
    return "\n".join(result)
