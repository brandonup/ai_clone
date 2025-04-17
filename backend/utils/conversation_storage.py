import os
import json
import logging
from google.cloud import firestore
import traceback

# --- Firestore Configuration ---
# Remove global initialization
# db = None
# CONVERSATIONS_COLLECTION = None
logger = logging.getLogger(__name__)

def _get_firestore_collection(collection_name='conversations'):
    """Helper to initialize client and get collection reference."""
    # This is duplicated from clone_manager, consider moving to a shared utils/db.py later
    try:
        db = firestore.Client()
        return db.collection(collection_name)
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client or get collection '{collection_name}': {e}", exc_info=True)
        return None

# --- Core Functions ---

def save_conversation(conversation_id: str, history: list):
    """Saves the conversation history to Firestore."""
    conversations_collection = _get_firestore_collection('conversations')
    if not conversations_collection:
        return False # Error already logged by helper
    if not conversation_id:
        logger.error("Cannot save conversation: Missing conversation_id.")
        return False

    try:
        doc_ref = conversations_collection.document(conversation_id)
        doc_ref.set({"messages": history})
        # logger.debug(f"Successfully saved conversation {conversation_id} to Firestore.")
        return True
    except Exception as e:
        logger.error(f"Error saving conversation {conversation_id} to Firestore: {e}", exc_info=True)
        return False

def load_conversation(conversation_id: str) -> list:
    """Loads the conversation history from Firestore."""
    conversations_collection = _get_firestore_collection('conversations')
    if not conversations_collection:
        return [] # Error already logged by helper
    if not conversation_id:
        logger.warning("load_conversation called with empty conversation_id.")
        return []

    try:
        doc_ref = conversations_collection.document(conversation_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            history = data.get("messages", [])
            if not isinstance(history, list):
                 logger.warning(f"Conversation {conversation_id} data['messages'] is not a list: {type(history)}. Returning empty list.")
                 return []
            # logger.debug(f"Successfully loaded conversation {conversation_id} from Firestore.")
            return history
        else:
            # logger.debug(f"Conversation {conversation_id} not found in Firestore. Returning empty list.")
            return []
    except Exception as e:
        logger.error(f"Error loading conversation {conversation_id} from Firestore: {e}", exc_info=True)
        return []

# --- Optional: Function to delete a conversation ---
def delete_conversation(conversation_id: str) -> bool:
    """Deletes a conversation document from Firestore."""
    conversations_collection = _get_firestore_collection('conversations')
    if not conversations_collection:
        return False # Error already logged by helper
    if not conversation_id:
        logger.warning("delete_conversation called with empty conversation_id.")
        return False
    try:
        doc_ref = conversations_collection.document(conversation_id)
        doc_ref.delete()
        logger.info(f"Successfully deleted conversation {conversation_id} from Firestore.")
        return True
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id} from Firestore: {e}", exc_info=True)
        return False
