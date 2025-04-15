"""
Conversation storage module for saving and loading conversation histories
"""
import os
import json
from datetime import datetime

# Directory for storing conversation histories
CONVERSATION_DIR = "conversations"
os.makedirs(CONVERSATION_DIR, exist_ok=True)

def save_conversation(conversation_id, history):
    """
    Save a conversation history to disk
    
    Args:
        conversation_id: Unique identifier for the conversation
        history: List of message objects
    """
    file_path = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=2)

def load_conversation(conversation_id):
    """
    Load a conversation history from disk
    
    Args:
        conversation_id: Unique identifier for the conversation
        
    Returns:
        list: Conversation history or empty list if not found
    """
    file_path = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []
