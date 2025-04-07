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
from utils.router import choose_route, generate_persona_patterns
from utils.router import persona_rag_patterns, persona_base_llm_patterns
from utils.llm_orchestrator import generate_answer, web_search, get_llm_for_entity_extraction
from utils.google_drive_utils import process_drive_folder
from utils.conversation_storage import save_conversation, load_conversation
from utils.memory_utils import ConversationBufferWindowMemory, format_conversation_history
from utils.entity_memory import ConversationEntityMemory

# Determine which RAG implementation to use
use_custom_rag = os.getenv("USE_CUSTOM_RAG", "false").lower() == "true"
if use_custom_rag:
    logger.info("Using custom RAG implementation with Small-to-Big chunking and sliding window")
    from utils.custom_rag_utils import ingest_document, retrieve_chunks
else:
    logger.info("Using Ragie.ai implementation for RAG")
    from utils.ragie_utils import ingest_document, retrieve_chunks

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
                
                # If using dev config, also generate persona patterns immediately
                try:
                    logger.info(f"Generating persona-specific patterns for DEV config: {coach_name}")
                    patterns = generate_persona_patterns(coach_name, coach_persona)
                    persona_rag_patterns = patterns.get('rag_patterns', [])
                    persona_base_llm_patterns = patterns.get('base_llm_patterns', [])
                    logger.info(f"Generated {len(persona_rag_patterns)} RAG patterns and {len(persona_base_llm_patterns)} base LLM patterns for DEV config")
                except Exception as e:
                    logger.error(f"Error generating persona-specific patterns for DEV config: {str(e)}")
                    persona_rag_patterns = []
                    persona_base_llm_patterns = []
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
        
        # Generate persona-specific patterns
        global persona_rag_patterns, persona_base_llm_patterns
        try:
            app.logger.info(f"Generating persona-specific patterns for {coach_name}")
            patterns = generate_persona_patterns(coach_name, coach_persona)
            persona_rag_patterns = patterns.get('rag_patterns', [])
            persona_base_llm_patterns = patterns.get('base_llm_patterns', [])
            app.logger.info(f"Generated {len(persona_rag_patterns)} RAG patterns and {len(persona_base_llm_patterns)} base LLM patterns")
        except Exception as e:
            app.logger.error(f"Error generating persona-specific patterns: {str(e)}")
            # Continue with empty patterns if generation fails
            persona_rag_patterns = []
            persona_base_llm_patterns = []
        
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
    
    # Initialize conversation history if it doesn't exist
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
    else:
        # Try to load conversation from disk
        disk_history = load_conversation(session['conversation_id'])
        if disk_history:
            # Only keep the last 10 messages in session to keep cookie size manageable
            if len(disk_history) > 10:
                session['conversation_history'] = disk_history[-10:]
            else:
                session['conversation_history'] = disk_history
    
    return render_template('chat_interface.html', conversation_history=session.get('conversation_history', []))

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
        logger.info(f"Processing question: {question}")
        
        # First, try to retrieve relevant chunks from the knowledge base
        chunks = retrieve_chunks(question, top_n=2)  # Reduced from 3 to 2 to limit token usage
        
        # Get the routing decision
        route = choose_route(question, chunks)
        logger.info(f"Router selected route: {route}")
        
        # Initialize prompt_composition
        prompt_composition = None
        
        # Get conversation memory (last 5 interactions)
        memory = ConversationBufferWindowMemory(k=5, conversation_id=session['conversation_id'])
        conversation_history = format_conversation_history(memory.get_memory_variables()["history"])
        
        # Get entity memory
        entity_memory = ConversationEntityMemory(
            conversation_id=session['conversation_id'],
            llm=get_llm_for_entity_extraction()
        )
        
        if chunks and route == "rag":
            # Use RAG with retrieved chunks
            source = "Knowledge Base (RAG)"
            context = "\n\n".join(chunks)
            answer, prompt_composition = generate_answer(
                question, 
                context=context, 
                coach_name=coach_name, 
                persona=coach_persona, 
                source=source,
                conversation_history=conversation_history,
                entity_memory=entity_memory
            )
            
        elif route == "force_rag":
            # Try a more direct retrieval with modified query
            modified_question = f"Find information about: {question}"
            logger.info(f"Trying modified query: {modified_question}")
            chunks = retrieve_chunks(modified_question, top_n=2)
            
            if chunks:
                source = "Knowledge Base (RAG)"
                context = "\n\n".join(chunks)
                answer, prompt_composition = generate_answer(
                    question, 
                    context=context, 
                    coach_name=coach_name, 
                    persona=coach_persona, 
                    source=source,
                    conversation_history=conversation_history,
                    entity_memory=entity_memory
                )
            else:
                # If no chunks found with modified query, use base LLM
                source = "Base LLM (No Knowledge Found)"
                answer, prompt_composition = generate_answer(
                    question, 
                    coach_name=coach_name, 
                    persona=coach_persona, 
                    source=source,
                    conversation_history=conversation_history,
                    entity_memory=entity_memory
                )
                
        elif route == "web":
            # Use web search for current information
            logger.info(f"Using web search for question: {question}")
            
            # Force web search to be used by adding explicit search terms
            enhanced_query = f"latest information about: {question}"
            logger.info(f"Enhanced web search query: {enhanced_query}")
            
            search_context = web_search(enhanced_query)
            
            if search_context:
                source = "Web Search (Serper.dev)"
                answer, prompt_composition = generate_answer(
                    question, 
                    context=search_context, 
                    coach_name=coach_name, 
                    persona=coach_persona, 
                    source=source,
                    conversation_history=conversation_history,
                    entity_memory=entity_memory
                )
                logger.info("Web search successful, using results for answer")
            else:
                # If web search fails, log detailed error and fall back to base LLM
                logger.error("Web search failed to return results")
                logger.info("Check if SERPAPI_API_KEY is valid and Serper.dev service is available")
                
                # Add a note about the web search failure to the answer
                source = "Base LLM (Web Search Failed)"
                base_answer, prompt_composition = generate_answer(
                    question, 
                    coach_name=coach_name, 
                    persona=coach_persona, 
                    source=source,
                    conversation_history=conversation_history,
                    entity_memory=entity_memory
                )
                
                # Add a note about the web search failure for time-sensitive questions
                if any(term in question.lower() for term in ["today", "yesterday", "tomorrow", "weather", "current", "latest"]):
                    answer = base_answer + "\n\n(Note: This answer was generated without access to current information. For real-time data like weather or current events, please check a dedicated service.)"
                else:
                    answer = base_answer
                
        else:  # route == "base"
            # Use base LLM for general knowledge questions
            logger.info(f"Using base LLM for question: {question}")
            source = "Base LLM"
            answer, prompt_composition = generate_answer(
                question, 
                coach_name=coach_name, 
                persona=coach_persona, 
                source=source,
                conversation_history=conversation_history,
                entity_memory=entity_memory
            )
        
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
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
    full_history.append({
        'role': 'assistant',
        'content': answer,
        'source': source,
        'prompt_composition': prompt_composition,  # Store prompt_composition
        'timestamp': timestamp
    })
    
    # Save the complete history to disk
    save_conversation(session['conversation_id'], full_history)
    
    # Update entity memory with the new interaction
    entity_memory.update_entity_from_interaction(question, answer)
    
    # For session, limit to last 10 messages to keep cookie size manageable
    if len(full_history) > 10:
        session['conversation_history'] = full_history[-10:]
    else:
        session['conversation_history'] = full_history
    
    return render_template('chat_interface.html', conversation_history=session['conversation_history'])

if __name__ == "__main__":
    app.run(debug=True, port=5005, use_reloader=True)
