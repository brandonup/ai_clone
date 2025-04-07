"""
Demo script for ConversationEntityMemory
"""
import os
import sys
import uuid
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the entity memory utilities
from utils.entity_memory import ConversationEntityMemory
from utils.llm_orchestrator import get_llm_for_entity_extraction

def print_separator():
    """Print a separator line"""
    print("\n" + "=" * 80 + "\n")

def demo_entity_memory():
    """
    Demonstrate the ConversationEntityMemory class
    """
    print("ConversationEntityMemory Demo")
    print("=============================")
    
    # Create a new conversation ID
    conversation_id = str(uuid.uuid4())
    print(f"Created new conversation with ID: {conversation_id}")
    
    # Create an entity memory instance
    try:
        llm = get_llm_for_entity_extraction()
        entity_memory = ConversationEntityMemory(conversation_id=conversation_id, llm=llm)
        print(f"Created entity memory")
    except Exception as e:
        print(f"Error creating entity memory: {str(e)}")
        print("Running demo without LLM capabilities")
        entity_memory = ConversationEntityMemory(conversation_id=conversation_id)
    
    # Simulate a conversation about startups
    conversations = [
        ("I'm struggling with finding product-market fit for my SaaS startup.", 
         "Finding product-market fit can be challenging. Can you tell me more about your startup and target market?"),
        
        ("We're targeting small accounting firms with our automated bookkeeping software. Our current CAC is $500 but our churn is high at 15% monthly.", 
         "That's valuable information. A 15% monthly churn rate is indeed high for a SaaS product targeting accounting firms. Have you gathered feedback from the customers who churned to understand why they're leaving?"),
        
        ("Yes, they say our software is missing some key features they need for tax preparation.", 
         "That's excellent feedback. It suggests your product isn't fully addressing the needs of your target market yet. For accounting firms, tax preparation features are likely a core requirement, not just a nice-to-have. This gap between what your product offers and what the market needs is exactly what product-market fit aims to address."),
        
        ("Should we pivot to a different market or add the tax features?", 
         "Before deciding between pivoting or adding features, I'd recommend a deeper analysis. Since you've already acquired customers and received specific feedback, adding tax preparation features might be the more efficient path. Small accounting firms clearly need your core bookkeeping solution, but the missing tax features are causing them to look elsewhere. What resources would it take to add these tax features?"),
        
        ("Our CTO John estimates it would take about 3 months with our current team of 5 developers.", 
         "Three months is a significant investment, but potentially worthwhile if it addresses the main reason for churn. Here's what I suggest: 1) Prioritize a minimum viable version of the tax features that addresses the most critical needs, 2) Consider reaching out to some churned customers to see if they'd return with these features, and 3) Meanwhile, improve onboarding for new customers to set proper expectations until the full feature set is ready. This approach lets you build toward product-market fit with your current target market, which seems promising given they're already adopting your core solution.")
    ]
    
    # Process each conversation turn
    print("\nProcessing conversation...")
    for i, (user_msg, ai_msg) in enumerate(conversations):
        print(f"\nExchange {i+1}:")
        print(f"User: {user_msg}")
        print(f"AI: {ai_msg}")
        
        # Update entity memory
        entity_memory.update_entity_from_interaction(user_msg, ai_msg)
        
        # Show entities after each exchange
        if entity_memory.entities:
            print("\nEntities identified so far:")
            for entity_name, entity_info in entity_memory.entities.items():
                print(f"- {entity_name} ({entity_info.get('type', 'UNKNOWN')})")
                if 'summary' in entity_info and entity_info['summary']:
                    print(f"  Summary: {entity_info['summary']}")
                if 'attributes' in entity_info and entity_info['attributes']:
                    print(f"  Attributes: {entity_info['attributes']}")
    
    print_separator()
    
    # Demonstrate entity retrieval
    test_queries = [
        "What was that issue with product-market fit?",
        "Tell me about the churn rate again",
        "Who was the CTO mentioned earlier?",
        "What was the timeline for adding new features?"
    ]
    
    print("Testing entity retrieval with queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        relevant_entities = entity_memory.get_relevant_entities(query)
        
        if relevant_entities:
            print("Relevant entities found:")
            for entity_name, entity_info in relevant_entities.items():
                print(f"- {entity_name} ({entity_info.get('type', 'UNKNOWN')})")
                if 'summary' in entity_info and entity_info['summary']:
                    print(f"  Summary: {entity_info['summary']}")
                
                # Show a sample of mentions
                mentions = entity_info.get('mentions', [])
                if mentions and len(mentions) > 0:
                    print(f"  Sample mention: {mentions[-1]}")
        else:
            print("No relevant entities found.")
    
    print_separator()
    
    # Demonstrate formatted entity memories
    print("Formatted entity memories for LLM context:")
    formatted_memories = entity_memory.format_entity_memories(entity_memory.entities)
    print(formatted_memories)
    
    print_separator()
    
    # Clean up
    print(f"Demo complete! Conversation saved to disk with ID: {conversation_id}")
    print("You can view this conversation's entities in entity_memories/{}.json".format(conversation_id))

if __name__ == "__main__":
    demo_entity_memory()
