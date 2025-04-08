"""
Main Flask application for AI Coach MVP with Google Drive integration
"""
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv
import os
import json
import traceback
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging to file
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_debug.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Import utility modules
# Removed old router imports: choose_route, generate_persona_patterns, persona_rag_patterns, persona_base_llm_patterns
from utils.llm_orchestrator import generate_answer # Removed web_search, get_llm_for_entity_extraction
from utils.google_drive_utils import process_drive_folder
from utils.conversation_storage import save_conversation, load_conversation
from utils.memory_utils import ConversationBufferWindowMemory, format_conversation_history
# Removed entity memory import: from utils.entity_memory import ConversationEntityMemory
from utils.adaptive_router import run_adaptive_rag # Import the new adaptive RAG function

# Determine which RAG implementation to use (for ingestion only now)
use_custom_rag = os.getenv("USE_CUSTOM_RAG", "false").lower() == "true"
if use_custom_rag:
    logger.info("Using custom RAG implementation for ingestion")
    from utils.custom_rag_utils import ingest_document # Removed retrieve_chunks
else:
    logger.info("Using Ragie.ai implementation for ingestion")
    from utils.ragie_utils import ingest_document # Removed retrieve_chunks

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Global variables to store coach information
# In a production app, these would be stored in a database
coach_name = None
coach_persona = None

# --- Development Setup Bypass ---
# Check for dev_config.json to bypass setup during development
# Note: We only check for dev_config.json, not dev_config_off.json
try:
    # Construct the path relative to the app.py file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'dev_config.json')
    
    try:
        with open(config_path, 'r') as f:
            dev_config = json.load(f)
            coach_name = dev_config.get("coach_name")
            coach_persona = dev_config.get("coach_persona")
            
            if coach_name and coach_persona:
                logger.warning("!!! Using coach name and persona from dev_config.json for development testing !!!")
                # Removed persona pattern generation as it's no longer used by the router
    except FileNotFoundError:
        logger.info("dev_config.json not found, proceeding with normal setup flow.")
except Exception as e:
    logger.error(f"Error reading or processing dev_config.json: {e}")
# --- End Development Setup Bypass ---


@app.route('/')
def index():
    """
    Root route - redirects to owner dashboard if setup not done,
    otherwise to chat interface
    """
    if coach_name is None or coach_persona is None:
        return redirect(url_for('owner_dashboard'))
    else:
        return redirect(url_for('chat'))

@app.route('/owner', methods=['GET', 'POST'])
def owner_dashboard():
    """
    Owner dashboard for Google Drive folder link and persona setup
    """
    global coach_name, coach_persona
    
    if request.method == 'POST':
        # Get coach's name
        name = request.form.get('coach_name')
        if not name:
            flash('Please provide a coach name')
            return redirect(url_for('owner_dashboard'))
        
        # Store coach's name
        coach_name = name
        
        # Get persona description
        persona = request.form.get('persona')
        if not persona:
            flash('Please provide a coach persona description')
            return redirect(url_for('owner_dashboard'))
        
        # Store persona
        coach_persona = persona
        
        # Removed persona pattern generation as it's no longer used by the router
        
        # Get Google Drive folder link (optional)
        drive_folder_url = request.form.get('drive_folder')
        
        # Only process Google Drive folder if a URL is provided and not empty
        if drive_folder_url and drive_folder_url.strip():
            try:
                # Process Google Drive folder
                app.logger.info(f"Processing Google Drive folder: {drive_folder_url}")
                files = process_drive_folder(drive_folder_url)
                
                if not files:
                    app.logger.warning("No supported files found in the Google Drive folder")
                    flash('No supported files found in the Google Drive folder, but setup will continue.')
                else:
                    # Ingest each file
                    for file in files:
                        try:
                            file_name = file.get('name')
                            file_content = file.get('content')
                            
                            # Ingest document
                            response = ingest_document(file_content, file_name)
                            
                            # Check response (optional)
                            if response:
                                app.logger.info(f"Document {file_name} uploaded successfully")
                            
                        except Exception as e:
                            app.logger.error(f"Error uploading {file_name}: {str(e)}")
                            flash(f"Error uploading {file_name}: {str(e)}")
                    
                    flash(f'Setup complete! {len(files)} documents indexed from Google Drive and persona profile saved.')
            except Exception as e:
                app.logger.error(f"Error processing Google Drive folder: {str(e)}")
                flash(f"Error processing Google Drive folder: {str(e)}, but setup will continue.")
        else:
            app.logger.info("No Google Drive folder URL provided, continuing with setup")
            flash('Setup complete! Coach profile saved successfully.')
        
        # Redirect to chat interface after setup is complete
        return redirect(url_for('chat'))
    
    return render_template('owner_dashboard.html')

@app.route('/chat', methods=['GET'])
def chat():
    """
    User chat interface
    """
    # Check if setup is complete
    if coach_name is None or coach_persona is None:
        flash('Coach setup is not complete. Please set up the coach first.')
        return redirect(url_for('owner_dashboard'))
    
    # Initialize conversation history for UI
    ui_history = [] # Initialize ui_history to an empty list

    if 'conversation_id' in session:
        # Always try to load the full history from disk for the UI
        disk_history = load_conversation(session['conversation_id'])
        ui_history = disk_history if disk_history else [] # Use full history for UI

        # Update the session (truncated) only if disk_history was loaded
        if disk_history:
            # Strip out prompt_composition to reduce cookie size
            session_history = []
            for msg in disk_history[-10:] if len(disk_history) > 10 else disk_history:
                # Create a copy of the message without prompt_composition
                session_msg = msg.copy()
                if 'prompt_composition' in session_msg:
                    del session_msg['prompt_composition']
                session_history.append(session_msg)
            
            session['conversation_history'] = session_history
        # If disk_history is empty/None, session['conversation_history'] might be outdated or empty

    else: # if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        # ui_history is already initialized to []

    return render_template('chat_interface.html', conversation_history=ui_history, coach_name=coach_name)

@app.route('/new_chat', methods=['GET'])
def new_chat():
    """
    Start a new chat conversation
    """
    # Create a new conversation ID and clear history
    session['conversation_id'] = str(uuid.uuid4())
    session['conversation_history'] = []
    
    return redirect(url_for('chat'))

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handle user questions and generate answers
    """
    # Check if setup is complete
    if coach_name is None or coach_persona is None:
        flash('Coach setup is not complete. Please set up the coach first.')
        return redirect(url_for('owner_dashboard'))
    
    # Get question from form
    question = request.form.get('question')
    if not question:
        flash('Please enter a question')
        return redirect(url_for('chat'))
    
    # Initialize conversation history if it doesn't exist
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Log the question for debugging
        logger.info(f"Processing question with Adaptive RAG: {question}")

        # Prepare LangSmith config (optional, based on environment variable)
        langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_config = {}
        if langsmith_api_key:
            langsmith_config = {
                "configurable": {
                    "metadata": {
                        "conversation_id": session['conversation_id'],
                        "coach_name": coach_name,
                    },
                    "run_name": f"AI Coach Adaptive RAG - {session['conversation_id']}",
                }
            }
            logger.info("LangSmith tracing enabled for Adaptive RAG.")

        # Get conversation history for context
        conversation_id = session['conversation_id']
        # Use ConversationBufferWindowMemory to get the last 5 interactions
        memory = ConversationBufferWindowMemory(k=5, conversation_id=conversation_id, return_messages=True) # Ensure it returns message objects
        # Get the windowed message objects (these are dictionaries with 'role' and 'content' keys)
        history_messages = memory.get_memory_variables().get(memory.memory_key, [])
        
        # Ensure history_messages is always a list, even if empty
        if history_messages is None:
            history_messages = []
            
        # Validate each message has the required format
        validated_messages = []
        for msg in history_messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                validated_messages.append(msg)
        
        # Format the windowed history for the preamble string (still needed for binding)
        conversation_history_str = format_conversation_history(validated_messages)
        
        # Debug log to verify we're only sending k=5 messages
        logger.info(f"Sending {len(validated_messages)} messages to the LLM")
        for msg in history_messages:
            logger.info(f"Message: {msg.get('role', 'unknown')}: {msg.get('content', '')[:50]}...")

        # --- Prompt Composition Explanation ---
        # ... (explanation omitted for brevity) ...
        # --- End Prompt Composition Explanation ---

        # Call the new adaptive RAG function with coach name, persona, and conversation history
        # This function handles routing, retrieval, web search, generation, and grading internally
        # Modified run_adaptive_rag call to handle potential early termination
        rag_result = {}
        try:
            # Use invoke instead of stream to get the final state directly
            # This avoids issues with stream termination causing KeyErrors
            final_state = run_adaptive_rag(
                question,
                coach_name=coach_name,
                persona=coach_persona,
                conversation_history=conversation_history_str,
                history_messages=validated_messages,
                config=langsmith_config
            )
            # Extract results from the final state dictionary
            rag_result = final_state if final_state else {}
            logger.info("Adaptive RAG finished successfully.")
        except Exception as graph_error:
            # Catch potential errors during graph execution (like recursion limit before fix)
            logger.error(f"Error during graph execution: {graph_error}")
            logger.error(traceback.format_exc())
            rag_result = {} # Ensure rag_result is defined even on error

        answer = rag_result.get("generation", "I'm sorry, I encountered an issue processing your request.")
        prompt_composition = rag_result.get("prompt_composition")
        documents = rag_result.get("documents", [])
        source_path = rag_result.get("source_path")

        # For debugging, log the document content
        if documents:
            logger.info(f"Retrieved {len(documents)} documents for RAG")
            for i, doc in enumerate(documents):
                 # Check if doc is a Document object before accessing page_content
                 if hasattr(doc, 'page_content'):
                     logger.info(f"Document {i+1} content: {doc.page_content[:200]}...")
                 else:
                     logger.info(f"Document {i+1} (not a Document object): {str(doc)[:200]}...")
        else:
            logger.info("No documents retrieved or used for RAG")

        # Determine the source based on the source_path from the graph
        if source_path == "web_search":
            source = "Web Search (Serper)"
        elif source_path == "base_llm" or source_path == "base_llm_cohere": # Handle both base_llm and base_llm_cohere
            source = "Base LLM"
        elif source_path == "vectorstore":
            source = "RAG (Knowledge Base)"
        elif answer and answer != "I'm sorry, I encountered an issue processing your request.":
            # If we have a valid answer but source_path is missing, assume it's from Base LLM
            # This handles the case where we fall back to base_llm after multiple generation attempts
            logger.info(f"Source path missing but valid answer received, assuming Base LLM")
            source = "Base LLM"
        else:
            # If source_path is None or unexpected, default to Error
            source = "Error"
            # If the answer is the default error message, keep source as Error
            if answer == "I'm sorry, I encountered an issue processing your request.":
                source = "Error"


    except Exception as e:
        error_msg = f"Error in /ask route before graph execution: {str(e)}"
        traceback_str = traceback.format_exc()
        app.logger.error(error_msg)
        logger.error(error_msg)
        logger.error(traceback_str)
        with open('error.log', 'a') as f:
            f.write(f"{error_msg}\n")
            f.write(f"{traceback_str}\n\n")
        answer = f"I'm sorry, I encountered an error while processing your question. Please try again later."
        source = "Error"
        prompt_composition = None
    
    # Load the full conversation history from disk (if it exists)
    full_history = load_conversation(session['conversation_id'])
    
    # If no history found on disk, use what's in the session
    if not full_history and 'conversation_history' in session:
        full_history = session.get('conversation_history', [])
    
    # Add the new messages to the full history
    full_history.append({
        'role': 'user',
        'content': question,
        'timestamp': timestamp
    })
    
    # Create message for disk storage (with prompt_composition)
    disk_message = {
        'role': 'assistant',
        'content': answer,
        'source': source,
        'timestamp': timestamp
    }
    
    # Only add prompt_composition to disk storage, not to session
    if prompt_composition:
        disk_message['prompt_composition'] = prompt_composition
        
    full_history.append(disk_message)
    
    # Save the complete history to disk
    save_conversation(session['conversation_id'], full_history)
    
    # Removed entity memory update
    # entity_memory.update_entity_from_interaction(question, answer)
    
    # For session, limit to last 10 messages to keep cookie size manageable
    # Also strip out prompt_composition to reduce cookie size
    session_history = []
    for msg in full_history[-10:] if len(full_history) > 10 else full_history:
        # Create a copy of the message without prompt_composition
        session_msg = msg.copy()
        if 'prompt_composition' in session_msg:
            del session_msg['prompt_composition']
        session_history.append(session_msg)
    
    session['conversation_history'] = session_history

    # Pass the full history (which was just updated and saved) to the template
    return render_template('chat_interface.html', conversation_history=full_history, coach_name=coach_name)

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=True)
