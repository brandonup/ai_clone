"""
Implementation for document retrieval and management using LangChain's Ragie integration
"""
import os
import logging
import json
import re
# Removed: from langchain_ragie import RagieRetriever
from langchain_core.documents import Document
from ragie import Ragie # Needed for ingestion, deletion, and now retrieval

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated function signature: clone_name instead of coach_name
def ingest_document(file_content, filename, clone_name, collection_name=None, delete_after_ingestion=True):
    """
    Ingest a document into Ragie.ai

    Args:
        file_content: File content as bytes or string.
        filename: Name of the file.
        clone_name: The name of the clone (used for partitioning).
        collection_name: The unique identifier for the clone's data (e.g., 'clone_uuid-...') used for partitioning and metadata.
        delete_after_ingestion: Whether to delete the local copy after ingestion.

    Returns:
        dict: Response from Ragie API
    """
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        raise ValueError("RAGIE_API_KEY environment variable not set")
    
    logger.info(f"Ingesting document: {filename}")
    
    # Convert to string if bytes and sanitize the text
    if isinstance(file_content, bytes):
        # Decode bytes to string
        text_content = file_content.decode('utf-8', errors='replace')
        # Convert back to bytes
        processed_content = text_content.encode('utf-8')
    else:
        # Convert to bytes
        processed_content = file_content.encode('utf-8')
    
    # Store a local copy of the document
    doc_path = None
    try:
        if isinstance(file_content, bytes):
            doc_text = file_content.decode('utf-8', errors='replace')
        else:
            doc_text = file_content
            
        doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents', filename)
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_text)
        logger.info(f"Stored local copy of document at {doc_path}")
    except Exception as e:
        logger.error(f"Error storing local copy of document: {str(e)}")
        doc_path = None
    
    with Ragie(auth=ragie_api_key) as r_client:
        # Include the collection_name (clone ID) in the metadata
        metadata = {
            "title": filename,
            "scope": "clone_data"
        }

        # Construct partition ID using clone_name and collection_name (identifier)
        partition_id = None
        if clone_name and collection_name:
            # Sanitize clone_name for partition ID (lowercase, alphanumeric, _, -)
            sanitized_clone_name = re.sub(r'[^a-z0-9_-]', '', clone_name.lower().replace(" ", "_"))
            # collection_name is already like clone_uuid-... which should be safe
            partition_id = f"{sanitized_clone_name}-{collection_name}"
            # Final check on the combined string (redundant but safe)
            partition_id = re.sub(r'[^a-z0-9_-]', '', partition_id)
            logger.info(f"Constructed partition ID: {partition_id}")
        else:
            logger.warning(f"Clone name ('{clone_name}') or collection identifier ('{collection_name}') missing. Document will go to default partition.")

        # Add collection_name (the unique identifier) as clone_id in metadata if provided
        if collection_name:
            metadata["clone_id"] = collection_name

        request_data = {
            "file": {
                "file_name": filename,
                "content": processed_content,
            },
            "metadata": metadata
        }

        # Add partition to the request if available
        if partition_id:
            request_data["partition"] = partition_id

        logger.info(f"Ragie ingest request data (metadata and partition): { {k: v for k, v in request_data.items() if k != 'file'} }") # Log without large file content

        try:
            response = r_client.documents.create(request=request_data)

            # Safely log response
            try:
                response_info = {
                    "status": "success",
                    "document_id": getattr(response, "id", "unknown"),
                    "document_type": str(type(response))
                }
                logger.info(f"Ragie ingest response info: {json.dumps(response_info, indent=2)}")
                
                # Delete the local file if requested
                if delete_after_ingestion and doc_path and os.path.exists(doc_path):
                    try:
                        os.remove(doc_path)
                        logger.info(f"Deleted local copy of document at {doc_path}")
                    except Exception as del_e:
                        logger.error(f"Error deleting local copy of document at {doc_path}: {str(del_e)}")
                
            except Exception as e:
                logger.info(f"Ragie ingest response: {str(response)}")
                
            return response
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise

def query_ragie(query: str, clone_id: str, top_k: int = 5) -> list:
    """
    Query Ragie.ai using the direct Ragie client to retrieve relevant documents,
    filtering by the mandatory clone_id metadata.

    Args:
        query (str): The user's query.
        clone_id (str): The mandatory clone ID to filter documents by metadata.
        top_k (int): The maximum number of documents to retrieve.

    Returns:
        list: A list of LangChain Document objects.
    """
    # Input validation
    if not query or not isinstance(query, str):
        logger.error(f"Invalid query: {query}")
        return []
    
    if not isinstance(top_k, int) or top_k <= 0:
        logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
        top_k = 5
    
    # Get API key
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("RAGIE_API_KEY environment variable not set for querying.")
        return []

    logger.info(f"Querying Ragie.ai directly: '{query}' (top_k={top_k}, clone_id={clone_id})")

    try:
        with Ragie(auth=ragie_api_key) as r_client:
            # Verify client has retrievals attribute and retrieve method
            if not hasattr(r_client, 'retrievals') or not hasattr(r_client.retrievals, 'retrieve'):
                logger.error("Ragie client doesn't support retrievals.retrieve method.")
                return []
            
            logger.info("Using Ragie retrievals.retrieve method")
            
            # Construct the retrieval request
            retrieval_request = {
                "query": query,
                "top_k": top_k
            }
            
            # Add mandatory metadata filter using the provided clone_id
            # This assumes 'clone_id' was added to metadata during ingestion
            retrieval_request["filter"] = {"clone_id": clone_id}
            logger.info(f"Applying mandatory filter: {retrieval_request['filter']}")

            # Call retrieve method
            retrieval_response = r_client.retrievals.retrieve(request=retrieval_request)
            
            documents = []
            # Process response
            if hasattr(retrieval_response, 'results') and retrieval_response.results:
                result_count = len(retrieval_response.results)
                logger.info(f"Retrieved {result_count} documents from Ragie")
                
                for result in retrieval_response.results:
                    # Extract content and metadata
                    content = getattr(result, 'text', '')
                    if not content:
                        logger.warning(f"Empty content in result, skipping. Result attributes: {dir(result)}")
                        continue
                    
                    # Build metadata
                    metadata = {}
                    # Add document_id to metadata if available
                    doc_id = getattr(result, 'id', None)
                    if doc_id:
                        metadata['document_id'] = doc_id
                    # Add score to metadata if available
                    score = getattr(result, 'score', None)
                    if score:
                        metadata['score'] = score
                    # Add source
                    metadata['source'] = 'ragie.ai'
                    # Add original metadata if available and is a dict
                    original_metadata = getattr(result, 'metadata', None)
                    if isinstance(original_metadata, dict):
                         metadata.update(original_metadata) # Merge original metadata

                    # Create LangChain Document
                    documents.append(Document(page_content=content, metadata=metadata))
                
                # Log a sample of the first document
                if documents:
                    sample_doc = documents[0]
                    logger.info(f"Sample document content: {sample_doc.page_content[:100]}...")
                    logger.info(f"Sample document metadata: {sample_doc.metadata}")
            else:
                logger.info("Ragie retrieval returned no results.")
                
            return documents
            
    except Exception as e:
        logger.error(f"Error querying Ragie.ai directly: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def delete_clone_documents(clone_id):
    """
    Delete all documents associated with a specific clone from Ragie.ai
    
    Args:
        clone_id: The ID of the clone whose documents should be deleted
        
    Returns:
        dict: Status of the deletion operation
    """
    logger.info(f"DELETE_DOCS: Function called with clone_id: {clone_id}")
    
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("DELETE_DOCS: RAGIE_API_KEY environment variable not set for document deletion")
        return {"status": "error", "message": "RAGIE_API_KEY not set"}
    
    if not clone_id:
        logger.warning("DELETE_DOCS: No clone ID provided, attempting to delete ALL documents")
        # Proceed to delete all documents if no clone_id is provided
    if not ragie_api_key:
        logger.error("DELETE_DOCS: RAGIE_API_KEY environment variable not set for document deletion")
        return {"status": "error", "message": "RAGIE_API_KEY not set"}
    
    logger.info(f"DELETE_DOCS: Attempting to delete all documents for clone: {clone_id}")
    
    try:
        logger.info(f"DELETE_DOCS: Creating Ragie client with API key: {ragie_api_key[:4]}...")
        with Ragie(auth=ragie_api_key) as r_client:
            logger.info(f"DELETE_DOCS: Ragie client created successfully")
            
            # Print all available attributes and methods for debugging
            logger.info(f"DELETE_DOCS: Available client attributes: {dir(r_client)}")
            
            # Check if we can access the documents collection
            if hasattr(r_client, 'documents'):
                logger.info(f"DELETE_DOCS: Available document methods: {dir(r_client.documents)}")
            
            # Try to list all documents directly
            try:
                # Check if list method exists
                if hasattr(r_client.documents, 'list'):
                    logger.info("DELETE_DOCS: Using documents.list() method")
                    documents_response = r_client.documents.list()
                    
                    if not documents_response or not hasattr(documents_response, 'documents'):
                        logger.info("DELETE_DOCS: No documents found in Ragie")
                        return {"status": "success", "message": "No documents found", "count": 0}
                    
                    documents = documents_response.documents
                    logger.info(f"DELETE_DOCS: Found {len(documents)} total documents")
                else:
                    logger.error("DELETE_DOCS: No list method found in documents")
                    return {"status": "error", "message": "No list method available in Ragie API"}
                
                # Check if we can see metadata in the results
                if documents:
                    sample_doc = documents[0]
                    logger.info(f"DELETE_DOCS: Sample document attributes: {dir(sample_doc)}")
                    
                    # Check if we can see document IDs
                    if hasattr(sample_doc, 'id'):
                        logger.info(f"DELETE_DOCS: Sample document ID: {sample_doc.id}")
                    
                    # Check if we can see metadata
                    if hasattr(sample_doc, 'metadata'):
                        logger.info(f"DELETE_DOCS: Sample metadata: {sample_doc.metadata}")
                
                # Try to find documents that match our clone ID
                matching_docs = []
                for doc in documents:
                    # Try different ways to access metadata
                    metadata = None
                    if hasattr(doc, 'metadata'):
                        metadata = doc.metadata
                    
                    # Log the metadata for debugging
                    logger.info(f"DELETE_DOCS: Document metadata: {metadata}")
                    
                    # Check if this document belongs to our clone
                    if metadata and isinstance(metadata, dict) and metadata.get('clone_id') == clone_id:
                        matching_docs.append(doc)
                    elif metadata and isinstance(metadata, dict) and metadata.get('scope') == 'clone_data':
                        # If clone_id is missing, but scope is clone_data, assume it belongs to this clone
                        logger.warning(f"DELETE_DOCS: Document missing clone_id, but has scope 'clone_data'. Assuming it belongs to clone {clone_id}")
                        matching_docs.append(doc)
                
                logger.info(f"DELETE_DOCS: Found {len(matching_docs)} documents matching clone_id {clone_id}")
                
                # If we found matching documents, try to delete them
                if matching_docs:
                    # Check if we have a delete method
                    if hasattr(r_client.documents, 'delete'):
                        deleted_count = 0
                        failed_count = 0
                        
                        for doc in matching_docs:
                            doc_id = getattr(doc, 'id', None)
                            
                            if not doc_id:
                                logger.warning(f"DELETE_DOCS: Document without ID found, skipping")
                                failed_count += 1
                                continue
                            
                            try:
                                logger.info(f"DELETE_DOCS: Attempting to delete document {doc_id}")
                                r_client.documents.delete(id=doc_id)
                                deleted_count += 1
                                logger.info(f"DELETE_DOCS: Successfully deleted document {doc_id}")
                            except Exception as del_e:
                                logger.error(f"DELETE_DOCS: Failed to delete document {doc_id}: {str(del_e)}")
                                failed_count += 1
                        
                        return {
                            "status": "success",
                            "message": f"Deleted {deleted_count} documents, failed to delete {failed_count}",
                            "deleted_count": deleted_count,
                            "failed_count": failed_count
                        }
                    else:
                        logger.error("DELETE_DOCS: No delete method found in Ragie client")
                        return {"status": "error", "message": "No delete method available in Ragie API"}
                else:
                    return {"status": "success", "message": "No matching documents found for this clone", "count": 0}
                
            except Exception as search_e:
                logger.error(f"DELETE_DOCS: Error searching documents: {str(search_e)}")
                import traceback
                logger.error(f"DELETE_DOCS: Traceback: {traceback.format_exc()}")
                return {"status": "error", "message": f"Error searching documents: {str(search_e)}"}
    
    except Exception as e:
        logger.error(f"DELETE_DOCS: Error deleting documents for clone {clone_id}: {str(e)}")
        import traceback
        logger.error(f"DELETE_DOCS: Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Error: {str(e)}"}
