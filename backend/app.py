"""
Backend Flask API for AI Clone MVP
"""
from flask import Flask, request, session, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import traceback
import uuid
from datetime import datetime
import threading
import logging
import importlib

from logging.handlers import RotatingFileHandler # Use RotatingFileHandler for better log management

# Load environment variables
load_dotenv()

# --- Initialize Flask app FIRST ---
app = Flask(__name__)

# --- Configure app.logger (writes to backend/app_flask.log) ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Corrected path: Create log file directly in the backend directory
file_handler = RotatingFileHandler('app_flask.log', maxBytes=1024*1024*5, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
if not app.logger.handlers: # Avoid adding duplicate handlers on reloads
    app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)
# --- End app.logger configuration ---

# --- Configure root logger (writes to backend/app_debug.log) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_debug.log', # Corrected path
    filemode='w'
)
logger = logging.getLogger(__name__) # Module-level logger for non-request-specific logs
# --- End root logger configuration ---

# --- Import utility modules with placeholders ---
enhance_persona = None
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
delete_clone_documents = None
delete_all_documents_ragie = None
ragie_utils = None

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
    use_custom_rag = os.getenv("USE_CUSTOM_RAG", "false").lower() == "true"
    if use_custom_rag:
        logger.info("Using custom RAG implementation for ingestion")
        from utils.custom_rag_utils import ingest_document # Assuming custom RAG doesn't need partition changes yet
    else:
        logger.info("Using Ragie.ai implementation")
        import utils.ragie_utils
        ragie_utils = utils.ragie_utils
        # Import functions needed later
        from utils.ragie_utils import ingest_document, delete_clone_documents, delete_all_documents_ragie

except ImportError as e:
    logger.error(f"Error importing utility modules: {e}. Ensure utils are in the backend directory.", exc_info=True)

# --- Configure Flask App Settings (after app is created) ---
# app = Flask(__name__) # Already created above
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_DOMAIN=None # Align with frontend for localhost development
)
CORS(app, supports_credentials=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# --- Helper Function ---
def get_active_clone_from_session():
    active_clone_id = session.get('active_clone_id')
    if not active_clone_id or not get_clone_by_id:
        return None
    return get_clone_by_id(active_clone_id)

# --- Development Setup Bypass ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'dev_config.json')
    with open(config_path, 'r') as f:
        dev_config = json.load(f)
        dev_clone_name = dev_config.get("clone_name")
        logger.warning("!!! dev_config.json found. It might influence clone data on first run. API relies on session['active_clone_id'] !!!")
        if load_clones:
            clones = load_clones()
            dev_clone_exists = any(c.get("clone_name") == dev_clone_name for c in clones)
            if not dev_clone_exists and dev_clone_name:
                 logger.info(f"Attempting to create dev clone '{dev_clone_name}' from dev_config.json if it doesn't exist.")
                 pass # Placeholder for dev creation logic
except FileNotFoundError:
    logger.info("dev_config.json not found, proceeding normally.")
except Exception as e:
    logger.error(f"Error reading or processing dev_config.json: {e}", exc_info=True)

# === API Endpoints ===

@app.route('/api/clones', methods=['GET'])
def get_clones_api():
    """API endpoint to get the list of all clones."""
    try:
        if not load_clones:
            return jsonify({"error": "Server configuration error (load_clones)."}), 500
        clones = load_clones()
        return jsonify(clones)
    except Exception as e:
        logger.error(f"Error loading clones: {e}", exc_info=True)
        return jsonify({"error": "Failed to load clone list."}), 500

@app.route('/api/clones/<clone_id>', methods=['GET'])
def get_clone_details_api(clone_id):
    """API endpoint to get details for a specific clone."""
    try:
        if not get_clone_by_id:
            return jsonify({"error": "Server configuration error (get_clone_by_id)."}), 500
        clone = get_clone_by_id(clone_id)
        if not clone:
            return jsonify({"error": "Clone not found"}), 404
        return jsonify(clone)
    except Exception as e:
        logger.error(f"Error getting clone details for {clone_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to get clone details."}), 500

@app.route('/api/clones', methods=['POST'])
def create_clone_api():
    """API endpoint to create a new clone."""
    logger.info("Entering create_clone_api function")
    try:
        data = request.json
        if not data:
            logger.error("Request data is missing or not JSON.")
            return jsonify({"error": "Invalid request data."}), 400
        logger.debug(f"Request data: {data}")

        # Extract data (coach_name removed)
        name = data.get('clone_name')
        original_persona = data.get('clone_persona')
        role = data.get('clone_role')
        clone_description = data.get('clone_description', '')
        drive_folder_url = data.get('drive_folder', '').strip()

        # Validation (coach_name removed)
        if not name or not original_persona or not role:
            missing_fields = [f for f in ['clone_name', 'clone_persona', 'clone_role'] if not data.get(f)]
            logger.error(f"Missing required fields in request: {missing_fields}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # --- Enhance Persona ---
        logger.info(f"Attempting to enhance persona for new clone: {name} with role: {role}")
        enhanced_persona_to_use = original_persona # Default
        if enhance_persona:
            try:
                enhanced_persona_result = enhance_persona(name, role, original_persona)
                if enhanced_persona_result:
                    enhanced_persona_to_use = enhanced_persona_result
                logger.info("Persona enhancement step completed.")
            except Exception as enhance_e:
                logger.error(f"Error during persona enhancement: {enhance_e}", exc_info=True)
                logger.warning("Proceeding with original persona due to enhancement error.")
        else:
             logger.warning("enhance_persona function not available.")

        category = role # Store role as category

        new_clone_id = str(uuid.uuid4())
        # This identifier is used for Ragie metadata AND forms part of the partition name
        ragie_collection_identifier = f"clone_{new_clone_id}"

        new_clone_data = {
            "id": new_clone_id,
            "clone_name": name,
            "clone_description": clone_description,
            "original_persona": original_persona,
            "enhanced_persona": enhanced_persona_to_use,
            "clone_persona": enhanced_persona_to_use,
            "clone_category": category,
            "clone_role": role,
            # "coach_name": coach_name, # Removed
            "drive_folder": drive_folder_url,
            "clone_id": ragie_collection_identifier # Store the identifier (Renamed from vectorstore_name)
        }
        logger.debug(f"Prepared new clone data: {new_clone_data}")

        # --- Determine initial ingestion status ---
        if drive_folder_url:
            new_clone_data['ingestion_status'] = 'pending'
            new_clone_data['files_processed'] = 0
            new_clone_data['files_attempted'] = 0
        else:
            new_clone_data['ingestion_status'] = 'complete'
        logger.info(f"Determined ingestion status: {new_clone_data['ingestion_status']}")

        # --- Add Clone to JSON ---
        logger.info(f"Attempting to add clone '{name}' to storage.")
        added_successfully = False
        try:
            if not add_clone:
                 logger.error("add_clone function is not available.")
                 return jsonify({"error": "Internal server configuration error (add_clone)."}), 500
            added_successfully = add_clone(new_clone_data)
        except Exception as add_e:
             logger.error(f"Error calling add_clone: {add_e}", exc_info=True)
             return jsonify({"error": "Internal server error during clone saving."}), 500

        if added_successfully:
            logger.info(f"Successfully added clone '{name}' with ID {new_clone_id} to storage.")
            # --- Start Background Ingestion ---
            if new_clone_data['ingestion_status'] == 'pending':
                logger.info(f"Starting background ingestion thread for clone {new_clone_id}")
                try:
                    if not background_ingestion_task:
                         logger.error("background_ingestion_task function not available.")
                    else:
                        # Pass the necessary data (no coach_name needed)
                        thread = threading.Thread(target=background_ingestion_task, args=(new_clone_data.copy(),))
                        thread.daemon = True
                        thread.start()
                        logger.info("Background ingestion thread started.")
                except Exception as thread_e:
                     logger.error(f"Failed to start background ingestion thread: {thread_e}", exc_info=True)
            return jsonify({"message": "Clone created successfully", "clone": new_clone_data}), 201
        else:
             logger.error(f"add_clone function returned False for clone '{name}'.")
             return jsonify({"error": "Failed to save clone profile (add_clone returned false)."}), 500

    except Exception as e: # Catch-all for the entire route
        logger.error(f"Unexpected error in create_clone_api: {e}", exc_info=True)
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

# --- Background Task for Document Ingestion ---
def background_ingestion_task(clone_data: dict):
    """Processes Google Drive folder and ingests documents in the background using Ragie."""
    ingestion_status = "error"
    error_message = "Task did not complete."
    files_processed = 0
    files_attempted = 0
    clone_id = clone_data.get("id", "UNKNOWN_ID")

    try:
        drive_folder_url = clone_data.get("drive_folder")
        ragie_collection_identifier = clone_data.get("clone_id") # Contains clone_id part (Renamed from vectorstore_name)
        clone_name = clone_data.get("clone_name", "Unknown")
        # coach_name = clone_data.get("coach_name") # Removed

        # Check if necessary functions are imported correctly
        if not process_drive_folder or not ingest_document:
             logger.error(f"Missing essential functions for background ingestion (clone: {clone_id}).")
             error_message = "Server configuration error: Missing ingestion functions."
             return # Use return to jump to finally

        # Validation (coach_name removed)
        if not drive_folder_url or not ragie_collection_identifier or not clone_id or not clone_name:
            logger.error(f"Missing data (drive_url, identifier, clone_id, clone_name) for background ingestion for clone {clone_id}")
            error_message = "Missing essential data for ingestion."
            return # Use return to jump to finally

        logger.info(f"Starting background ingestion for clone '{clone_name}' (ID: {clone_id}) using identifier '{ragie_collection_identifier}'")
        ingestion_status = "complete"
        error_message = None

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
                    if not file_name or file_content is None:
                         logger.warning(f"Skipping file with missing name or content in clone {clone_id}: {file.get('id', 'N/A')}")
                         continue

                    # Call ingest_document with clone_name instead of coach_name
                    response = ingest_document(
                        file_content=file_content,
                        filename=file_name,
                        clone_name=clone_name, # Pass clone_name
                        collection_name=ragie_collection_identifier, # Pass the identifier
                        delete_after_ingestion=True
                    )

                    ingestion_successful = False
                    if isinstance(response, dict) and response.get("status") == "success":
                         ingestion_successful = True
                    elif hasattr(response, 'id'):
                         ingestion_successful = True
                         logger.info(f"Background Task: Ragie response object received with ID: {getattr(response, 'id')} for file {file_name}")
                    else:
                         logger.warning(f"Background Task: Unexpected response type from ingest_document for file {file_name}: {type(response)}")

                    if ingestion_successful:
                        logger.info(f"Background Task: Document {file_name} uploaded successfully for clone {clone_id}")
                        files_processed += 1
                    else:
                         logger.error(f"Background Task: Failed to ingest document {file_name} for clone {clone_id}. Response: {response}")
                         ingestion_status = "error"
                         error_message = f"Error ingesting file: {file_name}. Check logs."
                         # break # Optional: stop on first error
                except Exception as e_file:
                    logger.error(f"Background Task: Error uploading file '{file_name}' for clone {clone_id}: {str(e_file)}", exc_info=True)
                    ingestion_status = "error"
                    error_message = f"Error processing file: {file_name}. Check logs."
                    # break # Optional: stop on first error
            logger.info(f"Background Task: Finished processing {files_attempted} files for clone {clone_id}. {files_processed} succeeded. Final Status: {ingestion_status}")

    except Exception as e_task:
        logger.error(f"Background Task: Error during ingestion process for clone {clone_id}: {str(e_task)}", exc_info=True)
        ingestion_status = "error"
        error_message = f"Error during ingestion task: {str(e_task)}"
        if 'files_attempted' not in locals(): files_attempted = 0
        if 'files_processed' not in locals(): files_processed = 0

    finally:
        # Update the clone status in Firestore
        try:
            if not load_clones or not save_clones:
                 logger.error(f"Cannot update clone status for {clone_id}: load/save functions unavailable.")
                 return

            clones = load_clones()
            updated = False
            for i, c in enumerate(clones):
                if c.get('id') == clone_id:
                    clones[i]['ingestion_status'] = ingestion_status
                    clones[i]['ingestion_error'] = error_message
                    clones[i]['files_processed'] = files_processed
                    clones[i]['files_attempted'] = files_attempted
                    save_clones(clones)
                    logger.info(f"Background Task: Updated final ingestion status to '{ingestion_status}' for clone {clone_id}")
                    updated = True
                    break
            if not updated: logger.error(f"Background Task: Could not find clone {clone_id} to update status.")
        except Exception as status_e: logger.error(f"Background Task: Failed to update final status for clone {clone_id}: {status_e}", exc_info=True)
# --- End Background Task ---


@app.route('/api/clones/<clone_id>', methods=['PUT'])
def manage_clone_api(clone_id):
    """API endpoint to update an existing clone."""
    try:
        if not get_clone_by_id or not update_clone:
            logger.error("get_clone_by_id or update_clone function not available.")
            return jsonify({"error": "Internal server configuration error (update_clone)."}), 500

        clone = get_clone_by_id(clone_id)
        if not clone:
            return jsonify({"error": "Clone not found"}), 404

        data = request.json
        if not data:
            return jsonify({"error": "Invalid request data."}), 400

        name = data.get('clone_name')
        original_persona = data.get('original_persona')
        role = data.get('clone_role')
        clone_description = data.get('clone_description', clone.get('clone_description'))
        drive_folder_url = data.get('drive_folder', clone.get('drive_folder')).strip()

        if not name or not original_persona or not role:
            missing_fields = [f for f in ['clone_name', 'original_persona', 'clone_role'] if not data.get(f)]
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # --- Re-enhance Persona if changed ---
        enhanced_persona_to_use = clone.get('enhanced_persona')
        persona_or_role_changed = (original_persona != clone.get('original_persona') or role != clone.get('clone_role'))

        if persona_or_role_changed and enhance_persona:
            logger.info(f"Persona or role changed for clone '{name}'. Re-enhancing.")
            try:
                enhanced_persona_result = enhance_persona(name, role, original_persona)
                if enhanced_persona_result:
                    enhanced_persona_to_use = enhanced_persona_result
                logger.info(f"Persona re-enhanced successfully for clone: {name}")
            except Exception as enhance_e:
                 logger.error(f"Error during persona re-enhancement: {enhance_e}", exc_info=True)
                 logger.warning("Proceeding with previously enhanced or original persona due to re-enhancement error.")
                 enhanced_persona_to_use = clone.get('enhanced_persona', original_persona) # Fallback
        else:
             logger.info(f"Original persona and role unchanged or enhance_persona unavailable for clone '{name}'.")

        category = role

        updated_data = {
            "clone_name": name,
            "clone_description": clone_description,
            "original_persona": original_persona,
            "enhanced_persona": enhanced_persona_to_use,
            "clone_persona": enhanced_persona_to_use,
            "clone_category": category,
            "clone_role": role,
            "drive_folder": drive_folder_url
        }

        if update_clone(clone_id, updated_data):
            updated_clone_data = get_clone_by_id(clone_id)
            return jsonify({"message": "Clone updated successfully", "clone": updated_clone_data})
        else:
             logger.error(f"update_clone returned False for clone {clone_id}")
             return jsonify({"error": "Error updating clone"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in manage_clone_api for {clone_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected internal server error occurred."}), 500


@app.route('/api/clone_status/<clone_id>', methods=['GET'])
def clone_status_api(clone_id):
    """API endpoint to check the ingestion status of a clone."""
    try:
        if not get_clone_by_id:
             logger.error("get_clone_by_id function not available.")
             return jsonify({"error": "Internal server configuration error (get_clone_by_id)."}), 500

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
    except Exception as e:
        logger.error(f"Error getting clone status for {clone_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to get clone status."}), 500


@app.route('/api/clones/<clone_id>', methods=['DELETE'])
def delete_clone_api(clone_id):
    """API endpoint to delete a clone."""
    try:
        if not get_clone_by_id or not delete_clone_data:
             logger.error("get_clone_by_id or delete_clone_data function not available.")
             return jsonify({"error": "Internal server configuration error (clone deletion)."}), 500

        clone = get_clone_by_id(clone_id)
        if not clone:
            return jsonify({"error": "Clone not found"}), 404

        ragie_collection_identifier = clone.get('clone_id') # Renamed from vectorstore_name
        logger.info(f"DELETE API: Found clone {clone_id} with Ragie identifier {ragie_collection_identifier}")

        # Delete documents from Ragie.ai
        deletion_result = {"status": "skipped", "message": "Document deletion skipped"}
        if ragie_collection_identifier:
            try:
                if delete_clone_documents:
                    logger.info(f"DELETE API: Calling delete_clone_documents for clone {clone_id} with identifier {ragie_collection_identifier}")
                    deletion_result = delete_clone_documents(ragie_collection_identifier)
                    logger.info(f"DELETE API: Document deletion result: {deletion_result}")
                else:
                     logger.error("DELETE API: delete_clone_documents function not available.")
                     deletion_result = {"status": "error", "message": "Deletion function not available."}
            except Exception as e:
                logger.error(f"DELETE API: Error deleting documents for clone {clone_id}: {str(e)}", exc_info=True)
                deletion_result = {"status": "error", "message": f"Error during deletion: {str(e)}"}
        else:
             logger.warning(f"DELETE API: No Ragie identifier found for clone {clone_id}. Skipping document deletion.")

        # Delete the clone data from Firestore
        if delete_clone_data(clone_id):
            logger.info(f"Successfully deleted clone data for {clone_id}")
            if session.get('active_clone_id') == clone_id:
                session.pop('active_clone_id', None)
                session.pop('conversation_id', None)
            return jsonify({
                "message": "Clone deleted successfully",
                "document_deletion": deletion_result
            })
        else:
            logger.error(f"delete_clone_data returned False for clone {clone_id}")
            return jsonify({"error": "Error deleting clone data", "document_deletion": deletion_result}), 500
    except Exception as e:
        logger.error(f"Unexpected error in delete_clone_api for {clone_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected internal server error occurred."}), 500


@app.route('/api/select_clone/<clone_id>', methods=['POST'])
def select_clone_api(clone_id):
    """API endpoint to select a clone for the session."""
    try:
        if not get_clone_by_id:
             logger.error("get_clone_by_id function not available.")
             return jsonify({"error": "Internal server configuration error (get_clone_by_id)."}), 500

        clone = get_clone_by_id(clone_id)
        if not clone:
            return jsonify({"error": "Clone not found"}), 404

        session['active_clone_id'] = clone_id
        new_conversation_id = str(uuid.uuid4())
        session['conversation_id'] = new_conversation_id
        logger.info(f"Selected clone {clone_id} and started new conversation {new_conversation_id}")

        return jsonify({
            "message": f"Clone {clone_id} selected",
            "clone_name": clone.get('clone_name'),
            "conversation_id": new_conversation_id
        })
    except Exception as e:
        logger.error(f"Error selecting clone {clone_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to select clone."}), 500


@app.route('/api/new_chat', methods=['POST'])
def new_chat_api():
    """API endpoint to start a new chat conversation with the currently selected clone."""
    try:
        if 'active_clone_id' not in session:
             return jsonify({"error": "No active clone selected"}), 400

        new_conversation_id = str(uuid.uuid4())
        session['conversation_id'] = new_conversation_id
        logger.info(f"Started new conversation {new_conversation_id} for clone {session['active_clone_id']}")

        return jsonify({"message": "New chat session started", "conversation_id": new_conversation_id})
    except Exception as e:
        logger.error(f"Error starting new chat: {e}", exc_info=True)
        return jsonify({"error": "Failed to start new chat."}), 500


@app.route('/api/conversation_history/<conversation_id>', methods=['GET'])
def get_conversation_history_api(conversation_id):
    """API endpoint to retrieve the history for a specific conversation."""
    logger.info(f"Request received for conversation history: {conversation_id}")
    try:
        if not load_conversation:
            logger.error("load_conversation function is not available.")
            return jsonify({"error": "Internal server configuration error (load_conversation)."}), 500

        history = load_conversation(conversation_id)
        if not isinstance(history, list):
            logger.warning(f"No valid history found or loaded for conversation {conversation_id}. Returning empty list.")
            history = [] # Ensure it's always a list

        logger.debug(f"Returning history for conv {conversation_id}: {json.dumps(history, indent=2)}")
        return jsonify({"conversation_history": history})

    except FileNotFoundError:
        logger.warning(f"Conversation file not found for ID: {conversation_id}. Returning empty history.")
        return jsonify({"conversation_history": []}) # Return empty list if file not found
    except Exception as e:
        logger.error(f"Error retrieving conversation history for {conversation_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve conversation history."}), 500


@app.route('/api/ask', methods=['POST'])
def ask_api():
    """API endpoint to handle user questions and generate answers."""
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request data."}), 400

    clone_id = data.get('clone_id')
    if not clone_id:
        clone_id = session.get('active_clone_id')
        if not clone_id:
            return jsonify({"error": "No clone ID provided or selected in session"}), 400

    if not get_clone_by_id:
        logger.error("get_clone_by_id function not available.")
        return jsonify({"error": "Internal server configuration error (get_clone_by_id)."}), 500

    active_clone = get_clone_by_id(clone_id)
    if not active_clone:
        if clone_id == session.get('active_clone_id'):
             session.pop('active_clone_id', None)
             session.pop('conversation_id', None)
        return jsonify({"error": "Active clone not found"}), 400

    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # First try to get conversation_id from the request data
    conversation_id = data.get('conversation_id')
    
    # If not in request, fall back to session
    if not conversation_id:
        session_conv_id = session.get('conversation_id')
        app.logger.info(f"/api/ask: Falling back to session conversation_id: {session_conv_id}")
        conversation_id = session_conv_id
        
    # If still no conversation_id, create a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        session['conversation_id'] = conversation_id
        logger.warning(f"Conversation ID missing for clone {clone_id}, created new one: {conversation_id}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    answer = "I'm sorry, I encountered an issue processing your request."
    source = "Error"
    prompt_composition = None
    full_history = []

    try:
        logger.info(f"Processing question for clone {active_clone['id']} conv {conversation_id}: {question}")

        langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_config = {}
        if langsmith_api_key:
            langsmith_config = {
                "configurable": {
                    "metadata": {"conversation_id": conversation_id, "clone_name": active_clone['clone_name']},
                    "run_name": f"AI Clone Adaptive RAG - {conversation_id}",
                }
            }

        history_messages = []
        conversation_history_str = ""
        if ConversationBufferWindowMemory and load_conversation and format_conversation_history:
            try:
                # Reduce memory window size to limit context length
                memory = ConversationBufferWindowMemory(k=3, conversation_id=conversation_id, return_messages=True)
                memory_vars = memory.get_memory_variables()
                # history_messages should contain BaseMessage objects if return_messages=True
                history_messages = memory_vars.get(memory.memory_key, [])
                # Validate message format before formatting (ensure they are BaseMessage or similar)
                # The validation in adaptive_router.py handles dict/BaseMessage conversion now.
                conversation_history_str = format_conversation_history(history_messages)
                logger.info(f"Retrieved {len(history_messages)} messages from memory for conversation {conversation_id}")
            except Exception as mem_e:
                 logger.error(f"Error retrieving/formatting conversation history: {mem_e}", exc_info=True)
                 history_messages = [] # Ensure history_messages is an empty list on error
                 conversation_history_str = "" # Ensure history string is empty on error
        else:
             logger.warning("Memory/History functions not available. Proceeding without history.")

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
                    clone_id=active_clone.get('clone_id'), # Renamed from vectorstore_name
                    conversation_history=conversation_history_str,
                    history_messages=history_messages,
                    config=langsmith_config
                )
                rag_result = final_state if final_state else {}
                logger.info("Adaptive RAG finished successfully.")
            except Exception as graph_error:
                logger.error(f"Error during graph execution: {graph_error}", exc_info=True)
                rag_result = {}
        else:
             logger.warning("run_adaptive_rag function not available.")

        answer = rag_result.get("generation", "I'm sorry, I encountered an issue processing your request.")
        prompt_composition = rag_result.get("prompt_composition")
        source_path = rag_result.get("source_path")

        if source_path == "web_search": source = "Web Search (Serper)"
        elif source_path == "base_llm" or source_path == "base_llm_cohere": source = "Base LLM"
        elif source_path == "vectorstore": source = "RAG (Knowledge Base)"
        elif answer and answer != "I'm sorry, I encountered an issue processing your request.": source = "Base LLM"
        else: source = "Error"
        if answer == "I'm sorry, I encountered an issue processing your request.": source = "Error"

    except Exception as e:
        error_msg = f"Error in /api/ask route: {str(e)}"
        logger.error(f"{error_msg}", exc_info=True)

    try:
        full_history = load_conversation(conversation_id) if load_conversation else []
        if not isinstance(full_history, list): full_history = []

        full_history.append({'role': 'user', 'content': question, 'timestamp': timestamp})
        disk_message = {'role': 'assistant', 'content': answer, 'source': source, 'timestamp': timestamp}
        if prompt_composition: disk_message['prompt_composition'] = prompt_composition
        full_history.append(disk_message)

        if save_conversation: save_conversation(conversation_id, full_history)
    except Exception as hist_e:
         logger.error(f"Error saving conversation history for {conversation_id}: {hist_e}", exc_info=True)

    # Log the history being returned for debugging
    logger.debug(f"Returning history for conv {conversation_id}: {json.dumps(full_history, indent=2)}")
    return jsonify({
        "answer": answer,
        "source": source,
        "conversation_history": full_history
    })

@app.route('/api/session_info', methods=['GET'])
def get_session_info():
    """API endpoint to get current session details."""
    try:
        active_clone_id = session.get('active_clone_id')
        conversation_id = session.get('conversation_id')
        active_clone_data = None
        if active_clone_id and get_clone_by_id:
             active_clone_data = get_clone_by_id(active_clone_id)

        return jsonify({
            "active_clone_id": active_clone_id,
            "conversation_id": conversation_id,
            "active_clone_name": active_clone_data.get('clone_name') if active_clone_data else None,
            "active_clone_role": active_clone_data.get('clone_role') if active_clone_data else None
        })
    except Exception as e:
        logger.error(f"Error getting session info: {e}", exc_info=True)
        return jsonify({"error": "Failed to get session info."}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Temporary route to delete all documents
@app.route('/api/delete_all_ragie_documents', methods=['POST'])
def delete_all_ragie_documents_api():
    """Temporary API endpoint to delete all documents from Ragie."""
    logger.warning("TEMPORARY ROUTE: Deleting all documents from Ragie")
    try:
        if delete_all_documents_ragie:
            result = delete_all_documents_ragie()
            logger.info(f"Deletion result: {result}")
            return jsonify({"message": "Attempted to delete all documents", "result": result})
        else:
            logger.error("delete_all_documents_ragie function not available.")
            return jsonify({"error": "Deletion function not available."}), 500
    except Exception as e:
        logger.error(f"Error deleting all documents: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask development server for backend API.")
    port = int(os.environ.get("PORT", 5003))
    # Use debug=False and potentially waitress/gunicorn in production
    app.run(debug=True, port=port, host='0.0.0.0', use_reloader=False)
