"""
Demonstration script for conversation memory features
This script shows how to work with stored conversations
"""
import os
import sys
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the conversation storage module
from utils.conversation_storage import (
    list_conversations,
    load_conversation,
    save_conversation,
    get_conversation_summary,
    load_conversation_with_metadata,
    save_conversation_with_metadata
)

def print_separator():
    """Print a separator line"""
    print("\n" + "=" * 80 + "\n")

def list_all_conversations():
    """List all saved conversations"""
    print("Available conversations:")
    conversations = list_conversations()
    
    if not conversations:
        print("  No conversations found.")
        return []
    
    for i, conv_id in enumerate(conversations):
        # Get summary if available
        summary = get_conversation_summary(conv_id)
        if summary:
            print(f"  {i+1}. ID: {conv_id}")
            print(f"     Messages: {summary['message_count']} ({summary['user_messages']} user, {summary['assistant_messages']} assistant)")
            print(f"     First message: {summary['first_message']}")
            print(f"     Last message: {summary['last_message']}")
        else:
            print(f"  {i+1}. ID: {conv_id} (No summary available)")
    
    return conversations

def view_conversation(conversation_id):
    """View a specific conversation"""
    history = load_conversation(conversation_id)
    
    if not history:
        print(f"Conversation {conversation_id} not found or empty.")
        return
    
    print(f"Conversation {conversation_id}:")
    print(f"Total messages: {len(history)}")
    print_separator()
    
    for msg in history:
        if msg['role'] == 'user':
            print(f"USER ({msg['timestamp']}):")
            print(f"  {msg['content']}")
        else:
            print(f"ASSISTANT ({msg['timestamp']}):")
            print(f"  {msg['content']}")
            if 'source' in msg:
                print(f"  Source: {msg['source']}")
        print()

def search_conversations(query):
    """
    Simple keyword search through conversations
    
    This is a basic implementation that could be enhanced with:
    1. Vector embeddings for semantic search
    2. Better relevance scoring
    3. Chunking for more granular results
    """
    print(f"Searching for: '{query}'")
    query = query.lower()
    results = []
    
    # Get all conversations
    conversations = list_conversations()
    
    for conv_id in conversations:
        history = load_conversation(conv_id)
        
        for i, msg in enumerate(history):
            if query in msg['content'].lower():
                # Get context (previous and next message if available)
                prev_msg = history[i-1] if i > 0 else None
                next_msg = history[i+1] if i < len(history)-1 else None
                
                results.append({
                    'conversation_id': conv_id,
                    'message': msg,
                    'prev_message': prev_msg,
                    'next_message': next_msg
                })
    
    # Print results
    if not results:
        print("No matches found.")
        return
    
    print(f"Found {len(results)} matches:")
    print_separator()
    
    for i, result in enumerate(results):
        print(f"Match {i+1} (Conversation: {result['conversation_id']}):")
        
        # Print previous message for context if available
        if result['prev_message']:
            prev = result['prev_message']
            print(f"PREVIOUS - {prev['role'].upper()} ({prev['timestamp']}):")
            print(f"  {prev['content'][:100]}..." if len(prev['content']) > 100 else f"  {prev['content']}")
            print()
        
        # Print the matching message
        msg = result['message']
        print(f"MATCH - {msg['role'].upper()} ({msg['timestamp']}):")
        print(f"  {msg['content']}")
        if 'source' in msg:
            print(f"  Source: {msg['source']}")
        print()
        
        # Print next message for context if available
        if result['next_message']:
            next_msg = result['next_message']
            print(f"NEXT - {next_msg['role'].upper()} ({next_msg['timestamp']}):")
            print(f"  {next_msg['content'][:100]}..." if len(next_msg['content']) > 100 else f"  {next_msg['content']}")
        
        if i < len(results) - 1:
            print_separator()

def main():
    """Main function"""
    print("AI Coach Conversation Memory Demo")
    print("=================================")
    
    while True:
        print("\nOptions:")
        print("1. List all conversations")
        print("2. View a specific conversation")
        print("3. Search conversations")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            list_all_conversations()
        
        elif choice == '2':
            conversations = list_all_conversations()
            if conversations:
                idx = input("\nEnter the number of the conversation to view (or 'b' to go back): ")
                if idx.lower() == 'b':
                    continue
                
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(conversations):
                        view_conversation(conversations[idx])
                    else:
                        print("Invalid conversation number.")
                except ValueError:
                    print("Please enter a valid number.")
        
        elif choice == '3':
            query = input("\nEnter search term: ")
            if query:
                search_conversations(query)
            else:
                print("Please enter a search term.")
        
        elif choice == '4':
            print("\nExiting. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
