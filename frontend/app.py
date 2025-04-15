"""
Frontend Flask application for AI Clone MVP
Serves HTML templates and static files.
Interacts with the backend API for dynamic data and actions.
"""
from flask import Flask, render_template, session, request, jsonify
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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-frontend") # Use a separate key or the same one

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

@app.route('/chat')
def chat_page():
    """Serves the main chat interface page."""
    # Chat history and clone details will be fetched/managed by JavaScript via API calls
    # Check if a clone is selected in the session (managed via JS/API now)
    # We might need an initial API call here or let JS handle fetching session info
    return render_template('chat_interface.html', backend_api_url=BACKEND_API_URL)

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
