"""
Demo script for ConversationBufferWindowMemory
"""
import os
import sys
import uuid
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the memory utilities
from utils.memory_utils import ConversationBufferWindowMemory, format_conversation_history

def print_separator():
    """Print a separator line"""
    print("\n" + "=" * 80 + "\n")

def demo_conversation_buffer_window_memory():
    """
    Demonstrate the ConversationBufferWindowMemory class
    """
    print("ConversationBufferWindowMemory Demo")
    print("==================================")
    
    # Create a new conversation ID
    conversation_id = str(uuid.uuid4())
    print(f"Created new conversation with ID: {conversation_id}")
    
    # Create a memory instance with window size 5
    memory = ConversationBufferWindowMemory(k=5, conversation_id=conversation_id)
    print(f"Created memory with window size k=5")
    
    # Simulate a conversation
    conversations = [
        ("Hello, I'm looking for advice on starting a business.", "Hi there! I'd be happy to help you with starting a business. What kind of business are you thinking about?"),
        ("I'm thinking about starting a software consulting company.", "That's a great choice! Software consulting can be lucrative and has relatively low startup costs. Do you have specific expertise or a target market in mind?"),
        ("I specialize in web development and want to target small businesses.", "Excellent! Small businesses often need web development help but can't afford large agencies. What specific services are you planning to offer?"),
        ("Mainly website development, e-commerce, and maybe some SEO.", "Those are all valuable services for small businesses. Have you thought about your pricing structure and how you'll market your services?"),
        ("Not yet, I'm still figuring that out. Any suggestions?", "For pricing, you could consider hourly rates, project-based pricing, or retainer models. As for marketing, I'd recommend starting with networking, creating a strong portfolio website, and perhaps content marketing to demonstrate your expertise."),
        ("That makes sense. How do I find my first clients?", "Finding your first clients can be challenging. Start with your existing network - former colleagues, friends in business, etc. Consider offering a discount for initial projects to build your portfolio. Local business groups and online platforms like Upwork can also be good starting points."),
        ("Should I create an LLC right away or start as a sole proprietor?", "This depends on your situation, but many consultants start as sole proprietors for simplicity and then transition to an LLC as they grow. An LLC provides liability protection and can have tax advantages, but comes with more paperwork and fees."),
        ("What about contracts? Do I need a lawyer?", "Having solid contracts is crucial. While you can start with templates from resources like the Freelancers Union, it's worth investing in a consultation with a business lawyer to customize them for your specific needs and ensure you're protected."),
    ]
    
    # Add messages to memory
    print("\nAdding conversation messages to memory...")
    for i, (user_msg, ai_msg) in enumerate(conversations):
        print(f"\nExchange {i+1}:")
        print(f"User: {user_msg}")
        print(f"AI: {ai_msg}")
        
        # Add messages to memory
        memory.add_user_message(user_msg)
        memory.add_ai_message(ai_msg, source="Base LLM")
        
        # Show current memory state
        if i % 2 == 1:  # Show every other exchange to keep output manageable
            print("\nCurrent memory state:")
            print(memory)
            
            # Get memory variables
            memory_vars = memory.get_memory_variables()
            print("\nMemory variables:")
            print(memory_vars["history"])
            
            # Show window sliding effect
            if i >= 5:
                print("\nNotice that the memory only keeps the last 5 interactions!")
    
    print_separator()
    
    # Demonstrate loading from disk
    print("Loading memory from disk...")
    new_memory = ConversationBufferWindowMemory(k=5, conversation_id=conversation_id)
    print("Memory loaded successfully!")
    
    # Show the loaded memory
    print("\nLoaded memory state:")
    print(new_memory)
    
    # Format as conversation history
    formatted_history = format_conversation_history(new_memory.get_memory_variables()["history"])
    print("\nFormatted conversation history (ready for LLM context):")
    print(formatted_history)
    
    print_separator()
    
    # Demonstrate different window sizes
    print("Demonstrating different window sizes:")
    
    for k in [2, 3, 5]:
        print(f"\nWindow size k={k}:")
        k_memory = ConversationBufferWindowMemory(k=k, conversation_id=conversation_id)
        print(k_memory)
    
    print_separator()
    
    # Clean up
    print(f"Demo complete! Conversation saved to disk with ID: {conversation_id}")
    print("You can view this conversation using the conversation_memory_demo.py script.")

if __name__ == "__main__":
    demo_conversation_buffer_window_memory()
