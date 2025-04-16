"""
Backend Flask API for AI Clone MVP
"""
from flask import Flask, request, session, jsonify # Removed: render_template, redirect, url_for, flash
from flask_cors import CORS # Added CORS
from dotenv import load_dotenv
import os
import json
import traceback
import uuid
from datetime import datetime
import threading # Added for background tasks
from qdrant_client import QdrantClient, models # Added Qdrant imports
from qdrant_client.http.models import Distance, VectorParams # Added Qdrant imports
import logging

# Load environment variables
load_dotenv()

# Set up logging to file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_debug.log', # Keep log file in backend directory
    filemode='w'
)
logger = logging.getLogger(__name__)

# Import utility modules (adjust paths if necessary after move)
try:
    from utils.llm_orchestrator import enhance_persona
    from utils.google_drive_utils import process_drive_folder
    from utils.conversation_storage import save_conversation, load_conversation
    from utils.memory_utils import ConversationBufferWindowMemory, format_conversation_history
    from utils.adaptive_router import run_adaptive_rag
    from utils.clone_manager import (
        load_clones, save_clones, get_clone_by_id,
        add_clone, update_clone, delete_clone_data,
        CATEGORY_TO_ROLE_MAP
    )
    # Determine which RAG implementation to use (for ingestion only now)
    use_custom_rag = os.getenv("USE_CUSTOM_RAG", "false").lower() == "true"
    if use_custom_rag:
        logger.info("Using custom RAG implementation for ingestion")
        from utils.custom_rag_utils import ingest_document
    else:
        logger.info("Using Ragie.ai implementation for ingestion")
        from utils.ragie_utils import ingest_document
except ImportError as e:
    logger.error(f"Error importing utility modules: {e}. Ensure utils are in the backend directory.")
    # Depending on the severity, you might want to exit or handle this differently
    enhance_persona = None # Define placeholders to avoid NameErrors later if imports fail
    process_drive_folder = None
    save_conversation = None
    load_conversation = None
    ConversationBufferWindowMemory = None
    format_conversation_history = None
    run_adaptive_rag = None
    load_clones = lambda: []
    save_clones = lambda _: None
    get_clone_by_id = lambda _: None
    add_clone = lambda _: False
    update_clone = lambda _, __: False
    delete_clone_data = lambda _: False
    CATEGORY_TO_ROLE_MAP = {}
    ingest_document = None


# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension

def create_new_qdrant_collection(collection_name: str) -> bool:
    """Creates a new Qdrant collection with the specified name."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("QDRANT_URL or QDRANT_API_KEY environment variables not set. Cannot create collection.")
        return False
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0) # Increased timeout
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if collection_name in collection_names:
             logger.warning(f"Qdrant collection '{collection_name}' already exists. Skipping creation.")
             return True
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
        )
        logger.info(f"Successfully created Qdrant collection: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error creating Qdrant collection '{collection_name}': {e}")
        return False

# Initialize Flask app
app = Flask(__name__)
# Add these lines for session cookie configuration:
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,  # Must be False for HTTP
    SESSION_COOKIE_HTTPONLY=True, # Good practice
    SESSION_COOKIE_DOMAIN='localhost' # <<< ADD THIS
)
CORS(app, supports_credentials=True) # Keep CORS after config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


# --- Helper Function to Get Active Clone ---
def get_active_clone_from_session():
    """Gets the active clone data based on the ID stored in the session."""
    active_clone_id = session.get('active_clone_id')
    if not active_clone_id:
        return None
    return get_clone_by_id(active_clone_id)

# --- Development Setup Bypass ---
# Check for dev_config.json to bypass setup during development
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'dev_config.json')
    with open(config_path, 'r') as f:
        dev_config = json.load(f)
        dev_clone_name = dev_config.get("clone_name")
        # ... (rest of dev config loading logic - kept for local dev) ...
        # This section primarily affects local testing and doesn't set a global state anymore.
        # It might pre-populate clones.json if needed for the first run.
        # The actual 'active' clone for API requests will depend on the session.
        logger.warning("!!! dev_config.json found. It might influence clones.json on first run. API relies on session['active_clone_id'] !!!")
        # Simplified dev logic: ensure the dev clone exists in clones.json
        clones = load_clones()
        dev_clone_exists = any(c.get("clone_name") == dev_clone_name for c in clones)
        if not dev_clone_exists and dev_clone_name:
             # Simplified creation logic if needed for dev
             logger.info(f"Attempting to create dev clone '{dev_clone_name}' from dev_config.json if it doesn't exist.")
             # ... (add simplified creation logic from original file if essential for dev setup) ...
             pass # Placeholder

except FileNotFoundError:
    logger.info("dev_config.json not found, proceeding normally.")
except Exception as e:
    logger.error(f"Error reading or processing dev_config.json: {e}")
# --- End Development Setup Bypass ---


# === API Endpoints ===

@app.route('/api/clones', methods=['GET'])
def get_clones():
    """API endpoint to get the list of all clones."""
    clones = load_clones()
    # Optionally filter/clean data before sending
    return jsonify(clones)

@app.route('/api/clones/<clone_id>', methods=['GET'])
def get_clone_details(clone_id):
    """API endpoint to get details for a specific clone."""
    clone = get_clone_by_id(clone_id)
    if not clone:
        return jsonify({"error": "Clone not found"}), 404
    # Optionally filter/clean data
    return jsonify(clone)


@app.route('/api/clones', methods=['POST'])
def create_clone_api():
    """API endpoint to create a new clone."""
    data = request.json
    name = data.get('clone_name')
    original_persona = data.get('clone_persona')
    role = data.get('clone_role') # Expecting role directly now
    clone_description = data.get('clone_description', '')
    drive_folder_url = data.get('drive_folder', '').strip()

    if not name or not original_persona or not role:
        return jsonify({"error": "Missing required fields: clone_name, clone_persona, clone_role"}), 400

    # --- Enhance Persona ---
    logger.info(f"Attempting to enhance persona for new clone: {name} with role: {role}")
    enhanced_persona = enhance_persona(name, role, original_persona) if enhance_persona else None
    enhanced_persona_to_use = enhanced_persona if enhanced_persona else original_persona

    category = role # Store role as category

    new_clone_id = str(uuid.uuid4())
    vectorstore_name = f"clone_{new_clone_id}" # Use the new UUID

    new_clone_data = {
        "id": new_clone_id,
        "clone_name": name,
        "clone_description": clone_description,
        "original_persona": original_persona,
        "enhanced_persona": enhanced_persona_to_use,
        "clone_persona": enhanced_persona_to_use,
        "clone_category": category,
        "clone_role": role,
        "drive_folder": drive_folder_url,
        "vectorstore_name": vectorstore_name
    }

    # --- Create Qdrant Collection ---
    collection_created = create_new_qdrant_collection(vectorstore_name)
    if not collection_created:
         return jsonify({"error": "Failed to create vector storage for the clone."}), 500

    # --- Determine initial ingestion status ---
    if drive_folder_url:
        new_clone_data['ingestion_status'] = 'pending'
        new_clone_data['files_processed'] = 0
        new_clone_data['files_attempted'] = 0
    else:
        new_clone_data['ingestion_status'] = 'complete'

    # --- Add Clone to JSON ---
    if add_clone(new_clone_data):
        logger.info(f"Successfully added clone '{name}' with ID {new_clone_id}")
        # --- Start Background Ingestion ---
        if new_clone_data['ingestion_status'] == 'pending':
            logger.info(f"Starting background ingestion thread for clone {new_clone_id}")
            thread = threading.Thread(target=background_ingestion_task, args=(new_clone_data.copy(),))
            thread.daemon = True
            thread.start()
        return jsonify({"message": "Clone created successfully", "clone": new_clone_data}), 201
    else:
         logger.error(f"Failed to add clone '{name}' to clones.json")
         # Consider deleting the Qdrant collection if saving fails
         return jsonify({"error": "Failed to save clone profile."}), 500


# --- Background Task for Document Ingestion (Keep as is) ---
def background_ingestion_task(clone_data: dict):
    """Processes Google Drive folder and ingests documents in the background."""
    drive_folder_url = clone_data.get("drive_folder")
    collection_name = clone_data.get("vectorstore_name")
    clone_id = clone_data.get("id")
    clone_name = clone_data.get("clone_name", "Unknown") # For logging

    if not drive_folder_url or not collection_name or not clone_id or not process_drive_folder or not ingest_document:
        logger.error(f"Missing data or functions for background ingestion for clone {clone_id}")
        # Update status to error
        try:
            clones = load_clones()
            for i, c in enumerate(clones):
                if c.get('id') == clone_id:
                    clones[i]['ingestion_status'] = 'error'
                    clones[i]['ingestion_error'] = 'Missing essential data or function for ingestion.'
                    save_clones(clones)
                    break
        except Exception as status_e:
             logger.error(f"Failed to update error status for clone {clone_id}: {status_e}")
        return

    logger.info(f"Starting background ingestion for clone '{clone_name}' (ID: {clone_id}) into collection '{collection_name}'")
    ingestion_status = "complete"
    error_message = None
    files_processed = 0
    files_attempted = 0

    try:
        logger.info(f"Background Task: Processing Google Drive folder: {drive_folder_url}")
        files = process_drive_folder(drive_folder_url)
        files_attempted = len(files) if files else 0

        if not files:
            logger.warning(f"Background Task: No supported files found for clone {clone_id}")
        else:
            logger.info(f"Background Task: Found {files_attempted} files for clone {clone_id}. Starting ingestion.")
            for file in files:
                try:
                    file_name = file.get('name')
                    file_content = file.get('content')
                    response = ingest_document(
                        file_content=file_content,
                        filename=file_name,
                        collection_name=collection_name,
                        delete_after_ingestion=True
                    )
                    if response and response.get("status") == "success":
                        logger.info(f"Background Task: Document {file_name} uploaded successfully for clone {clone_id}")
                        files_processed += 1
                    else:
                         logger.error(f"Background Task: Failed to ingest document {file_name} for clone {clone_id}. Response: {response}")
                         ingestion_status = "error"
                         error_message = f"Error ingesting file: {file_name}. Check logs."
                except Exception as e:
                    logger.error(f"Background Task: Error uploading {file_name} for clone {clone_id}: {str(e)}")
                    ingestion_status = "error"
                    error_message = f"Error processing file: {file_name}. Check logs."
            logger.info(f"Background Task: Finished processing {files_attempted} files for clone {clone_id}. {files_processed} succeeded. Status: {ingestion_status}")
    except Exception as e:
        logger.error(f"Background Task: Error processing Google Drive folder for clone {clone_id}: {str(e)}")
        ingestion_status = "error"
        error_message = f"Error processing Google Drive folder: {str(e)}"
    finally:
        # Update the clone status in clones.json
        try:
            clones = load_clones()
            updated = False
            for i, c in enumerate(clones):
                if c.get('id') == clone_id:
                    clones[i]['ingestion_status'] = ingestion_status
                    if error_message: clones[i]['ingestion_error'] = error_message
                    clones[i]['files_processed'] = files_processed
                    clones[i]['files_attempted'] = files_attempted
                    save_clones(clones)
                    logger.info(f"Background Task: Updated ingestion status to '{ingestion_status}' for clone {clone_id}")
                    updated = True
                    break
            if not updated: logger.error(f"Background Task: Could not find clone {clone_id} to update status.")
        except Exception as status_e: logger.error(f"Background Task: Failed to update final status for clone {clone_id}: {status_e}")
# --- End Background Task ---


@app.route('/api/clones/<clone_id>', methods=['PUT'])
def manage_clone_api(clone_id):
    """API endpoint to update an existing clone."""
    clone = get_clone_by_id(clone_id)
    if not clone:
        return jsonify({"error": "Clone not found"}), 404

    data = request.json
    name = data.get('clone_name')
    original_persona = data.get('original_persona')
    role = data.get('clone_role')
    clone_description = data.get('clone_description', clone.get('clone_description')) # Keep old if not provided
    drive_folder_url = data.get('drive_folder', clone.get('drive_folder')).strip() # Keep old if not provided

    if not name or not original_persona or not role:
        return jsonify({"error": "Missing required fields: clone_name, original_persona, clone_role"}), 400

    # --- Re-enhance Persona if changed ---
    enhanced_persona_to_use = clone.get('enhanced_persona')
    persona_or_role_changed = (original_persona != clone.get('original_persona') or role != clone.get('clone_role'))

    if persona_or_role_changed and enhance_persona:
        logger.info(f"Persona or role changed for clone '{name}'. Re-enhancing.")
        enhanced_persona = enhance_persona(name, role, original_persona)
        if not enhanced_persona:
            logger.warning(f"Persona re-enhancement failed for clone '{name}'. Using original.")
            enhanced_persona_to_use = original_persona
        else:
            logger.info(f"Persona re-enhanced successfully for clone: {name}")
            enhanced_persona_to_use = enhanced_persona
    else:
         logger.info(f"Original persona and role unchanged or enhance_persona unavailable for clone '{name}'.")

    category = role # Store role as category

    updated_data = {
        "clone_name": name,
        "clone_description": clone_description,
        "original_persona": original_persona,
        "enhanced_persona": enhanced_persona_to_use,
        "clone_persona": enhanced_persona_to_use,
        "clone_category": category,
        "clone_role": role,
        "drive_folder": drive_folder_url
        # Keep existing vectorstore_name, ingestion_status etc.
    }

    # Update the clone (update_clone should merge the dicts)
    if update_clone(clone_id, updated_data):
        updated_clone_data = get_clone_by_id(clone_id) # Get merged data
        return jsonify({"message": "Clone updated successfully", "clone": updated_clone_data})
    else:
         return jsonify({"error": "Error updating clone"}), 500


@app.route('/api/clone_status/<clone_id>', methods=['GET'])
def clone_status_api(clone_id):
    """API endpoint to check the ingestion status of a clone."""
    clone = get_clone_by_id(clone_id)
    if not clone:
        return jsonify({"error": "Clone not found"}), 404

    status = clone.get('ingestion_status', 'unknown')
    error_msg = clone.get('ingestion_error')
    files_processed = clone.get('files_processed')
    files_attempted = clone.get('files_attempted')

    response_data = {"status": status}
    if error_msg: response_data["error_message"] = error_msg
    if files_processed is not None: response_data["files_processed"] = files_processed
    if files_attempted is not None: response_data["files_attempted"] = files_attempted

    return jsonify(response_data)


@app.route('/api/clones/<clone_id>', methods=['DELETE'])
def delete_clone_api(clone_id):
    """API endpoint to delete a clone."""
    # Get the clone data before deletion to access vectorstore_name
    clone = get_clone_by_id(clone_id)
    if not clone:
        return jsonify({"error": "Clone not found"}), 404
    
    # Get the vectorstore_name (collection name) for this clone
    vectorstore_name = clone.get('vectorstore_name')
    logger.info(f"DELETE API: Found clone {clone_id} with vectorstore_name {vectorstore_name}")
    
    # Delete documents from Ragie.ai
    deletion_result = {"status": "skipped", "message": "Document deletion skipped"}
    try:
        # Import the delete_clone_documents function
        logger.info("DELETE API: Attempting to import delete_clone_documents function")
        try:
            from utils.ragie_utils import delete_clone_documents
            logger.info("DELETE API: Successfully imported delete_clone_documents function")
        except ImportError as ie:
            logger.error(f"DELETE API: ImportError when importing delete_clone_documents: {ie}")
            raise
        except Exception as e:
            logger.error(f"DELETE API: Unexpected error when importing delete_clone_documents: {e}")
            raise
        
        # Delete documents associated with this clone
        logger.info(f"DELETE API: Calling delete_clone_documents for clone {clone_id} with vectorstore_name {vectorstore_name}")
        deletion_result = delete_clone_documents(vectorstore_name)
        logger.info(f"DELETE API: Document deletion result: {deletion_result}")
    except ImportError as ie:
        logger.error(f"DELETE API: Could not import delete_clone_documents function: {ie}")
    except Exception as e:
        logger.error(f"DELETE API: Error deleting documents for clone {clone_id}: {str(e)}")
        import traceback
        logger.error(f"DELETE API: Traceback: {traceback.format_exc()}")
    
    # Delete the clone data
    if delete_clone_data(clone_id):
        # Clear active clone from session if it was the one deleted
        if session.get('active_clone_id') == clone_id:
            session.pop('active_clone_id', None)
            session.pop('conversation_id', None) # Also clear conversation
        
        # Return success with document deletion info
        return jsonify({
            "message": "Clone deleted successfully",
            "document_deletion": deletion_result
        })
    else:
        return jsonify({"error": "Error deleting clone", "document_deletion": deletion_result}), 500


@app.route('/api/select_clone/<clone_id>', methods=['POST'])
def select_clone_api(clone_id):
    """API endpoint to select a clone for the session."""
    clone = get_clone_by_id(clone_id)
    if not clone:
        return jsonify({"error": "Clone not found"}), 404

    # Set the active clone ID in the session
    session['active_clone_id'] = clone_id
    # Start a new conversation when selecting a clone
    new_conversation_id = str(uuid.uuid4()) # Store in a variable
    session['conversation_id'] = new_conversation_id
    logger.info(f"Selected clone {clone_id} and started new conversation {new_conversation_id}")

    # Return the new conversation ID along with other info
    return jsonify({
        "message": f"Clone {clone_id} selected",
        "clone_name": clone.get('clone_name'),
        "conversation_id": new_conversation_id # <<< ADD THIS
    })


@app.route('/api/new_chat', methods=['POST'])
def new_chat_api():
    """API endpoint to start a new chat conversation with the currently selected clone."""
    if 'active_clone_id' not in session:
         return jsonify({"error": "No active clone selected"}), 400

    # Create a new conversation ID and clear history associated with the old one (if any)
    session['conversation_id'] = str(uuid.uuid4())
    logger.info(f"Started new conversation {session['conversation_id']} for clone {session['active_clone_id']}")
    # No history to clear in session as it's loaded dynamically, but good practice

    return jsonify({"message": "New chat session started", "conversation_id": session['conversation_id']})


@app.route('/api/ask', methods=['POST'])
def ask_api():
    """API endpoint to handle user questions and generate answers."""
    data = request.json
    clone_id = data.get('clone_id') # <<< GET CLONE ID FROM REQUEST
    if not clone_id:
        return jsonify({"error": "No clone ID provided"}), 400

    active_clone = get_clone_by_id(clone_id) # Use clone_id from request
    if not active_clone:
        return jsonify({"error": "No active clone selected"}), 400

    question = data.get('question')
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Ensure conversation ID exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        logger.warning(f"Conversation ID missing, created new one: {session['conversation_id']}")

    conversation_id = session['conversation_id']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    answer = "I'm sorry, I encountered an issue processing your request."
    source = "Error"
    prompt_composition = None
    full_history = [] # Initialize

    try:
        logger.info(f"Processing question for clone {active_clone['id']} conv {conversation_id}: {question}")

        # Prepare LangSmith config
        langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_config = {}
        if langsmith_api_key:
            langsmith_config = {
                "configurable": {
                    "metadata": {"conversation_id": conversation_id, "clone_name": active_clone['clone_name']},
                    "run_name": f"AI Clone Adaptive RAG - {conversation_id}",
                }
            }

        # Get conversation history
        memory = ConversationBufferWindowMemory(k=5, conversation_id=conversation_id, return_messages=True) if ConversationBufferWindowMemory else None
        history_messages = memory.get_memory_variables().get(memory.memory_key, []) if memory else []
        validated_messages = [msg for msg in history_messages if isinstance(msg, dict) and 'role' in msg and 'content' in msg]
        conversation_history_str = format_conversation_history(validated_messages) if format_conversation_history else ""

        logger.info(f"Sending {len(validated_messages)} messages to the LLM")

        # Call adaptive RAG
        rag_result = {}
        if run_adaptive_rag:
            try:
                mapped_role = active_clone.get('clone_role', 'Assistant')
                logger.info(f"Calling adaptive RAG with persona: ... and mapped role: {mapped_role}")
                final_state = run_adaptive_rag(
                    question,
                    clone_name=active_clone['clone_name'],
                    clone_role=mapped_role,
                    persona=active_clone['clone_persona'],
                    vectorstore_name=active_clone.get('vectorstore_name'),
                    conversation_history=conversation_history_str,
                    history_messages=validated_messages,
                    config=langsmith_config
                )
                rag_result = final_state if final_state else {}
                logger.info("Adaptive RAG finished successfully.")
            except Exception as graph_error:
                logger.error(f"Error during graph execution: {graph_error}\n{traceback.format_exc()}")
                rag_result = {} # Ensure defined on error

        answer = rag_result.get("generation", "I'm sorry, I encountered an issue processing your request.")
        prompt_composition = rag_result.get("prompt_composition")
        documents = rag_result.get("documents", [])
        source_path = rag_result.get("source_path")

        # Determine source
        if source_path == "web_search": source = "Web Search (Serper)"
        elif source_path == "base_llm" or source_path == "base_llm_cohere": source = "Base LLM"
        elif source_path == "vectorstore": source = "RAG (Knowledge Base)"
        elif answer and answer != "I'm sorry, I encountered an issue processing your request.": source = "Base LLM" # Fallback assumption
        else: source = "Error"
        if answer == "I'm sorry, I encountered an issue processing your request.": source = "Error"

    except Exception as e:
        error_msg = f"Error in /api/ask route: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        # Keep default error answer/source

    # Load, update, and save conversation history
    full_history = load_conversation(conversation_id) if load_conversation else []
    if not full_history: full_history = [] # Ensure it's a list

    full_history.append({'role': 'user', 'content': question, 'timestamp': timestamp})
    disk_message = {'role': 'assistant', 'content': answer, 'source': source, 'timestamp': timestamp}
    if prompt_composition: disk_message['prompt_composition'] = prompt_composition
    full_history.append(disk_message)

    if save_conversation: save_conversation(conversation_id, full_history)

    # Return only the latest answer and source, plus the full history for the UI to update
    return jsonify({
        "answer": answer,
        "source": source,
        "conversation_history": full_history # Send full history back
    })

@app.route('/api/session_info', methods=['GET'])
def get_session_info():
    """API endpoint to get current session details."""
    active_clone_id = session.get('active_clone_id')
    conversation_id = session.get('conversation_id')
    active_clone_data = get_clone_by_id(active_clone_id) if active_clone_id else None

    return jsonify({
        "active_clone_id": active_clone_id,
        "conversation_id": conversation_id,
        "active_clone_name": active_clone_data.get('clone_name') if active_clone_data else None,
         # Add other relevant clone details if needed by frontend
        "active_clone_role": active_clone_data.get('clone_role') if active_clone_data else None
    })


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Use waitress or gunicorn for production, Flask dev server is fine for local
    # For Cloud Run, the entrypoint will be defined in the Dockerfile (e.g., using gunicorn)
    logger.info("Starting Flask development server for backend API.")
    app.run(debug=True, port=5001, host='0.0.0.0', use_reloader=False) # use_reloader=False is often better for stability inside containers
