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

def list_conversations():
    """
    List all saved conversations
    
    Returns:
        list: List of conversation IDs
    """
    conversations = []
    for filename in os.listdir(CONVERSATION_DIR):
        if filename.endswith('.json'):
            conversation_id = filename.replace('.json', '')
            conversations.append(conversation_id)
    return conversations

def save_conversation_with_metadata(conversation_id, history, metadata=None):
    """
    Save a conversation with metadata
    
    Args:
        conversation_id: Unique identifier for the conversation
        history: List of message objects
        metadata: Dictionary of metadata (optional)
    """
    if metadata is None:
        metadata = {}
    
    # Add default metadata if not provided
    if 'created_at' not in metadata:
        metadata['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'updated_at' not in metadata:
        metadata['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the data structure
    data = {
        'metadata': metadata,
        'history': history
    }
    
    # Save to disk
    file_path = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_conversation_with_metadata(conversation_id):
    """
    Load a conversation with metadata
    
    Args:
        conversation_id: Unique identifier for the conversation
        
    Returns:
        tuple: (history, metadata) or ([], {}) if not found
    """
    file_path = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('history', []), data.get('metadata', {})
    return [], {}

def delete_conversation(conversation_id):
    """
    Delete a conversation
    
    Args:
        conversation_id: Unique identifier for the conversation
        
    Returns:
        bool: True if deleted, False if not found
    """
    file_path = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def get_conversation_summary(conversation_id):
    """
    Get a basic summary of a conversation
    
    Args:
        conversation_id: Unique identifier for the conversation
        
    Returns:
        dict: Summary information or None if not found
    """
    history, metadata = load_conversation_with_metadata(conversation_id)
    if not history:
        return None
    
    # Count messages
    user_messages = sum(1 for msg in history if msg['role'] == 'user')
    assistant_messages = sum(1 for msg in history if msg['role'] == 'assistant')
    
    # Get first and last timestamps
    first_timestamp = history[0]['timestamp'] if history else None
    last_timestamp = history[-1]['timestamp'] if history else None
    
    return {
        'conversation_id': conversation_id,
        'message_count': len(history),
        'user_messages': user_messages,
        'assistant_messages': assistant_messages,
        'first_message': first_timestamp,
        'last_message': last_timestamp,
        'metadata': metadata
    }
