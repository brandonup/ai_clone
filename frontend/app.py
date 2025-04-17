"""
Frontend Flask application for AI Clone MVP
Serves HTML templates and static files.
Interacts with the backend API for dynamic data and actions.
"""
from flask import Flask, render_template, session, request, jsonify, redirect, url_for
from dotenv import load_dotenv
import os
import logging
import requests # To make requests to the backend API

# Load environment variables (optional for frontend, but good practice)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
# Ensure the secret key matches the backend's default if FLASK_SECRET_KEY is not set
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
# Reverted session cookie settings to Flask defaults
# app.config['SERVER_NAME'] = 'localhost:5002' # Explicitly set server name - REMOVED FOR SIMPLER LOCALHOST COOKIE HANDLING
app.config['SESSION_COOKIE_DOMAIN'] = None # Explicitly set cookie domain
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' # Explicitly match backend default

# Get Backend API URL from environment variable
# This MUST be set when running the frontend service
BACKEND_API_URL = os.getenv("BACKEND_API_URL")
if not BACKEND_API_URL:
    logger.warning("BACKEND_API_URL environment variable is not set. Frontend API calls will likely fail.")
    # In a real deployment, you might want to raise an error or exit
    # For local dev, you might default it e.g., "http://localhost:5001"


# --- Routes to Serve Pages ---

@app.route('/')
def index():
    """Redirect root to the clone library."""
    # In a pure SPA, this might serve the main index.html
    # For this multi-page Flask setup, redirecting is fine.
    # We need the backend URL available in the template context
    return render_template('clone_directory.html', backend_api_url=BACKEND_API_URL) # Renamed template

@app.route('/directory') # Renamed route
def clone_directory(): # Renamed function
    """Serves the clone directory page."""
    # The actual list of clones will be fetched by JavaScript using the backend API URL
    return render_template('clone_directory.html', backend_api_url=BACKEND_API_URL) # Renamed template

@app.route('/create')
def create_clone_page():
    """Serves the create clone page."""
    return render_template('create_clone.html', backend_api_url=BACKEND_API_URL)

@app.route('/manage/<clone_id>')
def manage_clone_page(clone_id):
    """Serves the manage clone (settings) page."""
    # Clone details will be fetched by JavaScript
    return render_template('clone_settings.html', clone_id=clone_id, backend_api_url=BACKEND_API_URL)

@app.route('/processing/<clone_id>')
def processing_clone_page(clone_id):
    """Serves the clone processing status page."""
    # Status will be checked by JavaScript
    # Pass clone_id needed by the template/JS
    return render_template('processing_clone.html', clone_id=clone_id, backend_api_url=BACKEND_API_URL)

@app.route('/chat') # No longer needs clone_id here
def chat_page():
    """Serves the main chat interface page."""
    # Get clone info from query parameters if available (passed from redirect)
    clone_id = request.args.get('clone_id')
    clone_name = request.args.get('clone_name')
    clone_role = request.args.get('clone_role') # Get role too
    conversation_id = request.args.get('conversation_id') # <<< GET CONVERSATION ID
    logger.info(f"Rendering chat page for clone_id={clone_id}, clone_name={clone_name}, clone_role={clone_role}, conversation_id={conversation_id}")
    # Pass these to the template, JS will use them for initial display
    return render_template(
        'chat_interface.html',
        backend_api_url=BACKEND_API_URL,
        initial_clone_id=clone_id,
        initial_clone_name=clone_name,
        initial_clone_role=clone_role,
        initial_conversation_id=conversation_id # <<< PASS CONVERSATION ID
    )

# --- NEW ROUTE: Select Clone and Chat ---
@app.route('/select-and-chat/<clone_id>')
def select_and_chat(clone_id):
    """Selects a clone and redirects to the chat page."""
    logger.info(f"Selecting clone {clone_id} and redirecting to chat")
    
    if not BACKEND_API_URL:
        logger.error("Backend API URL not configured")
        return "Backend API URL not configured", 500
    
    try:
        # Make a POST request to the backend API to select the clone
        response = requests.post(
            f"{BACKEND_API_URL}/api/select_clone/{clone_id}",
            headers={"Content-Type": "application/json"},
            cookies=request.cookies  # Forward cookies from the request
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            backend_response_data = response.json()
            clone_name = backend_response_data.get('clone_name', 'Unknown')
            conversation_id = backend_response_data.get('conversation_id') # <<< GET CONVERSATION ID

            # Fetch full clone details to get the role (Backend select response doesn't include it)
            clone_details_response = requests.get(f"{BACKEND_API_URL}/api/clones/{clone_id}", cookies=request.cookies)
            clone_role = None
            if clone_details_response.ok:
                clone_role = clone_details_response.json().get('clone_role')

            logger.info(f"Backend successfully selected clone {clone_id} (Name: {clone_name}, Role: {clone_role}, ConvID: {conversation_id})")
            # No longer need to set frontend session here, backend handles it.
            # Redirect to the chat page, passing info as query parameters
            logger.info(f"Attempting to redirect to chat page for clone {clone_id}")
            return redirect(url_for(
                'chat_page',
                clone_id=clone_id,
                clone_name=clone_name,
                clone_role=clone_role,
                conversation_id=conversation_id # <<< PASS CONVERSATION ID
            ))
        else:
            logger.error(f"Backend failed to select clone {clone_id}: {response.status_code} {response.text}")
            # Display a user-friendly error page or message instead of raw text
            # For now, keep the original error return for simplicity
            return f"Failed to select clone: {response.text}", 500
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error selecting clone {clone_id}: {e}")
        return f"Error selecting clone: {e}", 500

# --- Optional: Proxy Endpoint (Alternative to direct JS calls) ---
# You could create endpoints here that proxy requests to the backend.
# This can hide the backend URL from the browser and handle authentication centrally.
# Example (Not fully implemented here):
# @app.route('/api/clones', methods=['GET'])
# def proxy_get_clones():
#     if not BACKEND_API_URL:
#         return jsonify({"error": "Backend API URL not configured"}), 500
#     try:
#         resp = requests.get(f"{BACKEND_API_URL}/api/clones", cookies=request.cookies)
#         resp.raise_for_status()
#         return jsonify(resp.json())
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error proxying /api/clones: {e}")
#         return jsonify({"error": "Failed to fetch data from backend"}), 502


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # For Cloud Run, the entrypoint will be defined in the Dockerfile (e.g., using gunicorn)
    # Run on a different port than the backend (e.g., 5002)
    logger.info(f"Starting Flask development server for frontend. Backend API URL: {BACKEND_API_URL}")
    app.run(debug=True, port=5002, host='0.0.0.0', use_reloader=True) # Changed port to 5002
