# AI Coaching Clone MVP

This is an AI coaching application built with Flask, LangChain, and Ragie.ai. The application allows a coach to upload their content and define their persona, then provides an AI-powered interface for users to ask questions and receive responses in the coach's style.

## Features

- **Owner Dashboard**: Upload coaching documents and define the coach's persona
- **User Chat Interface**: Ask questions to the AI coach and receive personalized responses
- **Advanced RAG System**: 
  - **NEW**: Custom RAG implementation with Small-to-Big chunking and sliding window (see [CUSTOM_RAG.md](CUSTOM_RAG.md))
  - Optimized for token efficiency to handle large documents
  - Implements a RAG-first approach that prioritizes knowledge base retrieval
  - Includes fallback mechanisms for when token limits are exceeded
- **Intelligent Query Routing**: 
  - Uses Ragie.ai for knowledge base queries
  - Routes time-sensitive queries to web search for current information
  - Uses base LLM for general questions and advice
- **Web Search Integration**:
  - Automatically detects queries about current events, weather, etc.
  - Retrieves up-to-date information from the web when needed
  - Falls back gracefully when web search is unavailable
- **Consistent Coach Persona**: Maintains the coach's voice and style in all responses

## Project Structure

```
ai_coach_app/  
├── app.py                   # Flask application setup and route definitions  
├── requirements.txt         # Python dependencies  
├── .env.template            # Template for environment variables  
├── CUSTOM_RAG.md            # Documentation for custom RAG implementation
├── NGROK_SETUP.md           # Instructions for setting up ngrok
├── SERPAPI_SETUP.md         # Instructions for setting up SerpAPI
├── utils/                   # Utility modules for organization  
│   ├── ragie_utils.py       # Functions to ingest documents and retrieve data via Ragie  
│   ├── custom_rag_utils.py  # Custom RAG implementation with Small-to-Big chunking
│   ├── router.py            # Routing logic to decide Ragie vs Web vs Base LLM  
│   └── llm_orchestrator.py  # LangChain/LLM integration (prompt assembly & calls)  
├── templates/               # HTML templates for Flask  
│   ├── owner_dashboard.html # Upload form for Owner (content & persona setup)  
│   └── chat_interface.html  # Query interface for User (question input and answer display)  
└── static/                  # Static files (CSS, JS if needed)  
    └── style.css            # Basic styling for the web UI
```

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.template` and add your API keys:
   ```
   FLASK_ENV=development
   OPENAI_API_KEY=your-openai-key
   RAGIE_API_KEY=your-ragie-key
   SERPAPI_API_KEY=your-serpapi-key
   FLASK_SECRET_KEY=your-flask-secret-key
   
   # Custom RAG Configuration (optional)
   USE_CUSTOM_RAG=true
   QDRANT_URL=your-qdrant-url  # Optional, uses in-memory Qdrant if not provided
   QDRANT_API_KEY=your-qdrant-api-key  # Optional
   ```
4. Set up SerpAPI for web search functionality:
   - See [SERPAPI_SETUP.md](SERPAPI_SETUP.md) for detailed instructions
   - Web search is optional; the app will fall back to the base LLM if SerpAPI is not configured
5. Run the application:
   ```
   python app.py
   ```
6. Access the application:
   - Owner Dashboard: http://localhost:8080/owner
   - Chat Interface: http://localhost:8080/chat

## Remote Access with Ngrok

To make your locally running application accessible from any network:

1. Install ngrok if you haven't already: https://ngrok.com/download
2. Start the ngrok tunnel to your Flask app:
   ```
   ngrok http 8080
   ```
3. Share the generated ngrok URL with anyone who needs to access your application

For detailed instructions on using ngrok with this application, see [NGROK_SETUP.md](NGROK_SETUP.md).

## Usage

### For Coaches (Owners)

1. Visit the Owner Dashboard at `/owner`
2. Enter your coaching persona description (e.g., "Coach Jane is a productivity expert who gives friendly, concise advice.")
3. Upload your coaching documents (PDF, DOCX, TXT)
4. Submit the form to complete setup

### For Users

1. Visit the Chat Interface at `/chat`
2. Enter your question in the text area
3. Submit to receive a response from the AI coach

### Query Types and Routing

The AI Coach app intelligently routes different types of questions:

- **Knowledge Base Questions**: Questions about topics covered in your uploaded documents will be answered using the RAG system.
- **Time-Sensitive Questions**: Questions about current events, weather, or recent news will be routed to web search (if configured).
- **General Questions**: Questions about general topics or advice will be handled by the base language model.

### Examples of Time-Sensitive Questions

- "What's the weather forecast for tomorrow in New York?"
- "What happened in the stock market yesterday?"
- "Who won the game last night?"
- "What are the latest developments in AI technology?"

## Troubleshooting

### Web Search Not Working

If web search isn't working (indicated by "Base LLM (Web Search Failed)" in the source):

1. Check that your SerpAPI key is valid and not expired
2. Follow the instructions in [SERPAPI_SETUP.md](SERPAPI_SETUP.md) to update your key
3. Restart the application after updating the key

### Token Limit Exceeded

If you see a message about token limitations:

1. Try asking a more specific question
2. Break down complex questions into simpler ones
3. Consider uploading smaller, more focused documents

## Technologies Used

- **Flask**: Web framework for the application
- **LangChain**: For orchestrating LLM interactions
- **OpenAI API**: For generating responses and embeddings
- **Ragie.ai**: For document ingestion and semantic search
- **Qdrant**: Vector database for custom RAG implementation
- **SerpAPI**: For web search capabilities

## License

This project is for demonstration purposes only.
