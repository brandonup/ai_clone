import os
import json
import logging
from google.cloud import firestore
import traceback # For detailed error logging

# --- Firestore Configuration ---
# Remove global initialization
# db = None
# CLONES_COLLECTION = None
logger = logging.getLogger(__name__)

def _get_firestore_collection(collection_name='clones'):
    """Helper to initialize client and get collection reference."""
    try:
        # Explicitly set the project ID to ensure the correct one is used
        db = firestore.Client(project='prototype-app-456908')
        return db.collection(collection_name)
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client or get collection '{collection_name}': {e}", exc_info=True)
        return None

# --- Constants ---
# Removed CLONES_FILE_PATH

# Mapping from frontend category selection to internal role/persona guidance
CATEGORY_TO_ROLE_MAP = {
    "Characters": "Fictional Character",
    "Coaches": "Coach or Mentor",
    "Therapists": "Therapeutic Advisor",
    "Games": "Game Character or Assistant",
    "Historical": "Historical Figure",
    "Religion": "Religious Figure or Guide",
    "Animals": "Animal Persona",
    "Discussion": "Discussion Facilitator",
    "Comedy": "Comedian or Humorous Persona",
    "Other": "General Assistant",
    "Authors and books": "Author or Literary Character",
    "Celebrities": "Celebrity Persona",
    "Influencers": "Influencer Persona",
    "Companions": "Companion",
    "Romantic": "Romantic Partner",
    "Family": "Family Member",
    "Me!": "Personal Clone", # Special case?
    "Expert": "Subject Matter Expert",
    "Regular people": "Everyday Person",
    "CEO": "CEO Prsona",
    "Consultant": "Consultant Persona",
    "Spokesperson": "Spokesperson Persona"
}

# --- Core Functions ---

def load_clones() -> list:
    """Loads all clone profiles from Firestore."""
    clones_collection = _get_firestore_collection('clones')
    if not clones_collection:
        return [] # Error already logged by helper
    try:
        clones = []
        docs = clones_collection.stream()
        for doc in docs:
            clone_data = doc.to_dict()
            clone_data['id'] = doc.id # Ensure the document ID is included
            clones.append(clone_data)
        logger.info(f"Successfully loaded {len(clones)} clones from Firestore.")
        return clones
    except Exception as e:
        logger.error(f"Error loading clones from Firestore: {e}", exc_info=True)
        return []

def save_clones(clones: list):
    """
    Saves multiple clone profiles to Firestore.
    NOTE: This overwrites existing documents with the same ID.
    Consider using batch writes for efficiency if needed, but individual
    add/update operations are generally preferred from the API.
    """
    clones_collection = _get_firestore_collection('clones')
    if not clones_collection:
        return False # Error already logged by helper
    try:
        # This is less common now; usually add/update are used individually.
        # If batch saving is truly needed, implement using batch writes.
        logger.warning("save_clones function called. This might overwrite data unintentionally. Prefer add_clone/update_clone.")
        count = 0
        for clone_data in clones:
            clone_id = clone_data.get('id')
            if clone_id:
                # Use set() which creates or overwrites the document
                clones_collection.document(clone_id).set(clone_data)
                count += 1
            else:
                logger.warning(f"Skipping clone data without ID: {clone_data.get('clone_name', 'N/A')}")
        logger.info(f"Attempted to save/overwrite {count} clones in Firestore.")
        return True # Indicate attempt was made
    except Exception as e:
        logger.error(f"Error saving clones to Firestore: {e}", exc_info=True)
        return False

def get_clone_by_id(clone_id: str) -> dict | None:
    """Retrieves a specific clone profile by its ID from Firestore."""
    clones_collection = _get_firestore_collection('clones')
    if not clones_collection:
        return None # Error already logged by helper
    if not clone_id:
        logger.warning("get_clone_by_id called with empty clone_id.")
        return None
    try:
        logger.info(f"Attempting to get clone by ID: {clone_id}") # Log the ID being requested
        doc_ref = clones_collection.document(clone_id)
        logger.debug(f"Document reference path: {doc_ref.path}")
        doc = doc_ref.get()
        logger.info(f"Firestore doc.exists for ID {clone_id}: {doc.exists}") # Log existence check result
        if doc.exists:
            clone_data = doc.to_dict()
            clone_data['id'] = doc.id # Ensure ID is included
            # logger.info(f"Found clone with ID: {clone_id}") # Can be noisy
            return clone_data
        else:
            logger.warning(f"Clone with ID {clone_id} not found in Firestore.")
            return None
    except Exception as e:
        logger.error(f"Error getting clone {clone_id} from Firestore: {e}", exc_info=True)
        return None

def add_clone(clone_data: dict) -> bool:
    """Adds a new clone profile to Firestore."""
    clones_collection = _get_firestore_collection('clones')
    if not clones_collection:
        return False # Error already logged by helper
    clone_id = clone_data.get('id')
    if not clone_id:
        logger.error("Cannot add clone: Missing 'id' in clone data.")
        return False
    try:
        # Check if document already exists (optional, set overwrites anyway)
        # doc_ref = CLONES_COLLECTION.document(clone_id)
        # if doc_ref.get().exists:
        #     logger.warning(f"Clone with ID {clone_id} already exists. Overwriting.")

        # Use set() to create the document with the specific ID
        clones_collection.document(clone_id).set(clone_data)
        logger.info(f"Successfully added clone '{clone_data.get('clone_name', 'N/A')}' with ID {clone_id} to Firestore.")
        return True
    except Exception as e:
        logger.error(f"Error adding clone {clone_id} to Firestore: {e}", exc_info=True)
        return False

def update_clone(clone_id: str, update_data: dict) -> bool:
    """Updates specific fields of an existing clone profile in Firestore."""
    clones_collection = _get_firestore_collection('clones')
    if not clones_collection:
        return False # Error already logged by helper
    if not clone_id:
        logger.warning("update_clone called with empty clone_id.")
        return False
    try:
        doc_ref = clones_collection.document(clone_id)
        # Use update() to modify specific fields without overwriting the whole doc
        # Note: update() will fail if the document doesn't exist.
        # If you want "upsert" behavior (create if not exists, update if exists),
        # you might use doc_ref.set(update_data, merge=True) instead.
        # Let's stick to update() for now, assuming the clone exists.
        doc_ref.update(update_data)
        logger.info(f"Successfully updated clone with ID {clone_id} in Firestore.")
        return True
    except firestore.NotFound:
         logger.error(f"Error updating clone: Document with ID {clone_id} not found.")
         return False
    except Exception as e:
        logger.error(f"Error updating clone {clone_id} in Firestore: {e}", exc_info=True)
        return False

def delete_clone_data(clone_id: str) -> bool:
    """Deletes a clone profile from Firestore by its ID."""
    clones_collection = _get_firestore_collection('clones')
    if not clones_collection:
        return False # Error already logged by helper
    if not clone_id:
        logger.warning("delete_clone_data called with empty clone_id.")
        return False
    try:
        doc_ref = clones_collection.document(clone_id)
        doc_ref.delete()
        logger.info(f"Successfully deleted clone data with ID: {clone_id} from Firestore.")
        return True
    except Exception as e:
        logger.error(f"Error deleting clone {clone_id} from Firestore: {e}", exc_info=True)
        return False

# --- Initialization (Load clones on startup - less critical now) ---
# Initial load might still be useful for some checks or caching, but primary operations hit DB.
# logger.info("Initial load of clones from Firestore...")
# clones_list = load_clones()
# logger.info(f"Loaded {len(clones_list)} clones initially.")
