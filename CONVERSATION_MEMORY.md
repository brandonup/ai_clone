# Conversation Memory for AI Coach

This document explains the conversation memory feature implemented in the AI Coach application and how it can be extended in the future for retrieval capabilities.

## Current Implementation

The AI Coach application now stores complete conversation histories to disk, enabling future memory and retrieval features. The implementation includes:

### 1. Conversation Storage Module

The `utils/conversation_storage.py` module provides functions for:

- Saving conversations to disk
- Loading conversations from disk
- Listing all available conversations
- Adding metadata to conversations
- Getting conversation summaries
- Deleting conversations

### 2. Storage Format

Each conversation is stored as a JSON file in the `conversations/` directory, with the conversation ID as the filename. The basic format is an array of message objects:

```json
[
  {
    "role": "user",
    "content": "User's question",
    "timestamp": "YYYY-MM-DD HH:MM:SS"
  },
  {
    "role": "assistant",
    "content": "AI Coach's response",
    "source": "Source of information (e.g., Knowledge Base, Web Search)",
    "timestamp": "YYYY-MM-DD HH:MM:SS"
  },
  ...
]
```

For conversations with metadata, the format is:

```json
{
  "metadata": {
    "created_at": "YYYY-MM-DD HH:MM:SS",
    "updated_at": "YYYY-MM-DD HH:MM:SS",
    "additional_field": "value"
  },
  "history": [
    {
      "role": "user",
      "content": "User's question",
      "timestamp": "YYYY-MM-DD HH:MM:SS"
    },
    ...
  ]
}
```

### 3. Integration with Flask App

The Flask application has been updated to:

- Save the complete conversation history to disk after each exchange
- Load conversation history from disk when resuming a session
- Keep only the most recent 10 messages in the session cookie to manage size
- Create a new conversation ID for each new chat

### 4. Demo Script

A demonstration script (`utils/conversation_memory_demo.py`) is provided to show how to:

- List all saved conversations
- View the contents of a specific conversation
- Perform basic keyword searches across conversations

## Future Retrieval Capabilities

The current implementation lays the groundwork for more advanced memory and retrieval features. Here are some ways to extend the system:

### 1. Basic Keyword Search

The simplest approach is to implement keyword-based search across conversation histories, as demonstrated in the demo script. This can be enhanced with:

- Better relevance scoring
- Fuzzy matching for typos
- Filtering by date ranges or other metadata
- Highlighting matching terms in results

### 2. Vector Embeddings for Semantic Search

For more advanced retrieval, implement semantic search using vector embeddings:

```python
# Example implementation (pseudocode)
def create_embeddings(text):
    # Use OpenAI or other embedding model
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def index_conversations():
    conversations = list_conversations()
    for conv_id in conversations:
        history = load_conversation(conv_id)
        for msg in history:
            # Create embedding for each message
            embedding = create_embeddings(msg['content'])
            # Store embedding with message ID
            store_embedding(conv_id, msg, embedding)

def semantic_search(query):
    # Create embedding for query
    query_embedding = create_embeddings(query)
    # Find closest matches
    results = find_similar_embeddings(query_embedding)
    return results
```

### 3. Chunking for More Granular Retrieval

Instead of treating each message as a unit, chunk conversations into smaller pieces:

```python
def chunk_conversation(history, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(history), chunk_size - overlap):
        chunk = history[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
```

### 4. Hybrid Approach

Combine multiple retrieval methods for better results:

1. Use keyword search to filter initial candidates
2. Use semantic search to rank by relevance
3. Use metadata to filter by time, topic, etc.
4. Use chunking to retrieve the most relevant parts of conversations

### 5. Memory Integration with LLM

Modify the `generate_answer` function in `utils/llm_orchestrator.py` to include relevant conversation history:

```python
def generate_answer(question, context=None, coach_name="AI Coach", 
                   persona="Persona", source="Base LLM", 
                   conversation_history=None, memory_results=None):
    # ... existing code ...
    
    # Add memory results to system message if available
    if memory_results:
        system_content += "\n=== RELEVANT PAST CONVERSATIONS ===\n"
        for result in memory_results:
            system_content += f"User previously asked: {result['question']}\n"
            system_content += f"You answered: {result['answer']}\n\n"
    
    # ... rest of existing code ...
```

## Implementation Roadmap

To implement full conversation memory retrieval, follow this roadmap:

1. **Phase 1: Basic Memory (Implemented)**
   - Store complete conversation histories
   - Provide basic tools for accessing stored conversations

2. **Phase 2: Simple Retrieval**
   - Implement keyword search across conversations
   - Add a memory route in the router
   - Modify the `/ask` route to handle memory queries

3. **Phase 3: Advanced Retrieval**
   - Implement vector embeddings for semantic search
   - Add chunking for more granular retrieval
   - Create a database for storing embeddings

4. **Phase 4: Memory Integration**
   - Include relevant memory in all responses
   - Add conversation summarization
   - Implement memory management (pruning, archiving)

## Usage Examples

### Example 1: Explicit Memory Query

When a user asks about a previous conversation:

```
User: "What did you tell me about product-market fit last week?"
```

The system would:
1. Detect this as a memory query
2. Search conversation history for "product-market fit"
3. Retrieve relevant past exchanges
4. Generate a response that references the previous conversation

### Example 2: Implicit Memory Usage

Even for regular questions, the system could use memory to provide more personalized responses:

```
User: "I'm still struggling with my startup idea."
```

The system would:
1. Process as a normal question
2. Also search memory for relevant context about the user's startup
3. Include this context when generating the response
4. Create a more personalized answer that builds on previous conversations

## Conclusion

The conversation memory feature provides a foundation for more advanced AI coaching capabilities. By storing complete conversation histories, the AI Coach can build a more personalized relationship with users, reference past discussions, and provide more contextually relevant advice.
