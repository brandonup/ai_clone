# Conversation Storage Directory

This directory stores conversation histories for the AI Coach application. Each conversation is saved as a JSON file with the conversation ID as the filename.

## File Format

Each conversation file contains an array of message objects with the following structure:

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

## Usage

The conversation history is automatically saved to disk when a user asks a question. The full history is stored on disk, while only the most recent 10 messages are kept in the session to keep cookie size manageable.

This storage mechanism allows for future implementation of conversation memory features, such as:

1. Searching through past conversations
2. Referencing previous discussions
3. Analyzing conversation patterns
4. Building a personalized knowledge base for each user

## Note

These files contain sensitive conversation data. In a production environment, consider:

1. Implementing proper access controls
2. Adding encryption for sensitive data
3. Setting up regular backups
4. Implementing data retention policies
