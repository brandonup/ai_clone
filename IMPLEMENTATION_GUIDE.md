# AI Coaching Clone - Implementation Guide

## Overview
This document provides a detailed guide to the AI Coaching Clone application implementation. The application allows coaches to upload their content, define their persona, and then provides an AI-powered interface for users to ask questions and receive responses in the coach's style.

## Architecture
The application follows a modular architecture with clear separation of concerns:

1. **Web Interface (Flask)**: Handles HTTP requests, renders templates, and manages user interactions
2. **Document Management (Ragie.ai)**: Handles document ingestion, indexing, and retrieval
3. **Query Routing**: Determines how to answer questions (knowledge base, web search, or base LLM)
4. **LLM Orchestration**: Manages prompt construction and LLM interactions

## Key Components

### 1. Flask Application (app.py)
The main application file handles routes, form submissions, and orchestrates the overall flow:
- `/owner`: Dashboard for document upload and persona setup
- `/chat`: Interface for asking questions
- `/ask`: Endpoint for processing questions and generating answers

### 2. Utility Modules
- **ragie_utils.py**: Functions for document ingestion and retrieval via Ragie.ai
- **router.py**: Logic for deciding how to answer queries (Ragie vs Web vs Base LLM)
- **llm_orchestrator.py**: LangChain integration for generating responses with the coach's persona

### 3. User Interfaces
- **owner_dashboard.html**: Form for uploading documents and setting the coach persona
- **chat_interface.html**: Interface for asking questions to the AI coach
- **style.css**: Styling for the web interface

## Implementation Details

### Document Ingestion
Documents are uploaded through the owner dashboard and sent to Ragie.ai for processing:
```python
# Read file as binary
file_content = file.read()
                
# Ingest document to Ragie - pass raw bytes directly
response = ingest_document(file_content, file.filename)
```

### Query Routing
When a user asks a question, the application determines the best way to answer:
```python
# Retrieve chunks from Ragie
chunks = retrieve_chunks(question)
        
# Determine route
route = choose_route(question, chunks)
```

### Response Generation
Based on the route, the application generates an answer using the appropriate context:
```python
if route == "ragie":
    context = "\n\n".join(chunks)
    answer = generate_answer(question, context=context, persona=coach_persona)
elif route == "web":
    search_context = web_search(question)
    answer = generate_answer(question, context=search_context, persona=coach_persona)
else:  # "base"
    answer = generate_answer(question, persona=coach_persona)
```

## Technical Considerations

### API Keys
The application requires three API keys:
- OpenAI API key for LLM access
- Ragie API key for document management
- SerpAPI key for web search

These are stored in a .env file and loaded at runtime.

### Error Handling
The application includes error handling for API failures and invalid inputs:
```python
try:
    # API calls and processing
except Exception as e:
    app.logger.error(f"Error generating answer: {str(e)}")
    answer = f"I'm sorry, I encountered an error while processing your question. Please try again later."
```

### Security Considerations
- API keys are stored in environment variables, not in code
- File uploads are limited to 16MB
- Input validation is performed before processing

## Deployment
For production deployment:
1. Clone the repository to your server
2. Install dependencies using `pip install -r requirements.txt`
3. Set up your .env file with your API keys
4. Run the application with a production WSGI server (e.g., Gunicorn)

## Future Enhancements
Potential improvements for future versions:
1. User authentication for multiple coaches
2. Conversation history
3. More sophisticated routing logic
4. Upgrade to GPT-4 for improved responses
5. Analytics to track usage patterns
