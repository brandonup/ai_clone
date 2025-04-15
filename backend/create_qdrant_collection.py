"""
Script to create the Qdrant collection for AI Coach
"""
import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "clone_docs_d7ba27c0-a08b-4462-a732-2e36ffebd6e2"
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension

def create_qdrant_collection():
    """
    Create the Qdrant collection for AI Coach
    """
    # Load environment variables
    load_dotenv()
    
    # Get Qdrant URL and API key from environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL or QDRANT_API_KEY environment variables not set")
        return False
    
    logger.info(f"Connecting to Qdrant at {qdrant_url}")
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            logger.info(f"Qdrant collection already exists: {COLLECTION_NAME}")
            return True
        
        # Create collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_qdrant_collection()
    
    if success:
        print(f"Successfully created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"Failed to create Qdrant collection: {COLLECTION_NAME}")
