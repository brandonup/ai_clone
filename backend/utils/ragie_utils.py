"""
Implementation for document retrieval and management using LangChain's Ragie integration
"""
import os
import logging
import json
import re
import time
from langchain_ragie import RagieRetriever
from langchain_core.documents import Document
from ragie import Ragie  # Still needed for document ingestion and deletion

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

def query_ragie(query: str, top_k: int = 5, clone_id: str = None, max_retries: int = 3) -> list:
    """
    Query Ragie.ai to retrieve relevant documents with improved robustness.

    Args:
        query (str): The user's query.
        top_k (int): The maximum number of documents to retrieve.
        clone_id (str): Optional clone ID to filter documents by.
        max_retries (int): Maximum number of retry attempts for transient errors.

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

    logger.info(f"Querying Ragie.ai: '{query}' (top_k={top_k}, clone_id={clone_id})")

    # Initialize retry counter
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # If this is a retry, add exponential backoff
            if retry_count > 0:
                backoff_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8... seconds
                logger.info(f"Retry attempt {retry_count}/{max_retries}, waiting {backoff_time} seconds...")
                time.sleep(backoff_time)
            
            # Try direct Ragie API first
            with Ragie(auth=ragie_api_key) as r_client:
                # Verify client has retrievals attribute and retrieve method
                if not hasattr(r_client, 'retrievals') or not hasattr(r_client.retrievals, 'retrieve'):
                    logger.warning("Ragie client doesn't have retrievals.retrieve method, falling back to LangChain")
                    break  # Break out of retry loop and fall back to LangChain
                
                logger.info("Using Ragie retrievals.retrieve method")
                
                # Try multiple filtering strategies
                documents = []
                filter_strategies = []
                
                # Strategy 1: Filter by clone_id if provided
                if clone_id:
                    filter_strategies.append({
                        "name": "clone_id direct filter",
                        "filter": {"clone_id": clone_id}
                    })
                
                # Strategy 2: Filter by metadata.clone_id if clone_id provided
                if clone_id:
                    filter_strategies.append({
                        "name": "metadata.clone_id filter",
                        "filter": {"metadata.clone_id": clone_id}
                    })
                
                # Strategy 3: Filter by scope=clone_data
                filter_strategies.append({
                    "name": "scope filter",
                    "filter": {"scope": "clone_data"}
                })
                
                # Strategy 4: No filter (fallback)
                filter_strategies.append({
                    "name": "no filter",
                    "filter": None
                })
                
                # Try each filter strategy until we get results
                for strategy in filter_strategies:
                    strategy_name = strategy["name"]
                    filter_value = strategy["filter"]
                    
                    # Create retrieval request
                    retrieval_request = {
                        "query": query,
                        "top_k": top_k
                    }
                    
                    # Add filter if specified
                    if filter_value:
                        retrieval_request["filter"] = filter_value
                        logger.info(f"Trying filter strategy: {strategy_name} with filter: {filter_value}")
                    else:
                        logger.info(f"Trying filter strategy: {strategy_name} (no filter)")
                    
                    try:
                        # Call retrieve method
                        retrieval_response = r_client.retrievals.retrieve(request=retrieval_request)
                        
                        # Process response
                        if hasattr(retrieval_response, 'results') and retrieval_response.results:
                            result_count = len(retrieval_response.results)
                            logger.info(f"✅ Strategy '{strategy_name}' successful! Retrieved {result_count} documents")
                            
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
                                
                                # Add filter strategy that worked
                                metadata['filter_strategy'] = strategy_name
                                metadata['source'] = 'ragie.ai'
                                
                                # Create LangChain Document
                                documents.append(Document(page_content=content, metadata=metadata))
                            
                            # If we got results, no need to try other strategies
                            if documents:
                                break
                        else:
                            logger.info(f"Strategy '{strategy_name}' returned no results")
                    
                    except Exception as strategy_e:
                        logger.warning(f"Error with filter strategy '{strategy_name}': {str(strategy_e)}")
                        continue  # Try next strategy
                
                # If we got documents from any strategy, return them
                if documents:
                    logger.info(f"Successfully retrieved {len(documents)} documents from Ragie")
                    # Log a sample of the first document
                    if len(documents) > 0:
                        sample_doc = documents[0]
                        logger.info(f"Sample document content: {sample_doc.page_content[:100]}...")
                        logger.info(f"Sample document metadata: {sample_doc.metadata}")
                    return documents
                
                # If we tried all strategies and got no results, log it
                logger.warning("All filter strategies failed to retrieve documents")
                
                # Fall back to LangChain integration
                logger.info("Falling back to LangChain RagieRetriever")
                break  # Break out of retry loop to try LangChain
            
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(f"Error querying Ragie (attempt {retry_count}/{max_retries}): {str(e)}")
            else:
                logger.error(f"Failed to query Ragie after {max_retries} attempts: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                break  # Break out of retry loop to try LangChain
    
    # Fall back to LangChain integration
    try:
        logger.info("Using LangChain RagieRetriever as fallback")
        retriever = RagieRetriever(
            api_key=ragie_api_key,
            top_k=top_k
        )
        
        # Use invoke instead of get_relevant_documents (which is deprecated)
        documents = retriever.invoke(query)
        
        if documents:
            logger.info(f"Received {len(documents)} results from Ragie via LangChain.")
            # Log a sample of the first document
            if len(documents) > 0:
                sample_doc = documents[0]
                logger.info(f"Sample document content: {sample_doc.page_content[:100]}...")
                logger.info(f"Sample document metadata: {sample_doc.metadata}")
        else:
            logger.info("No results received from Ragie.")
            
        return documents

    except Exception as e:
        logger.error(f"Error querying Ragie.ai via LangChain: {str(e)}")
        import traceback
        traceback.print_exc()
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

def delete_all_documents_ragie():
    """
    Deletes all documents from Ragie.ai.
    
    Returns:
        dict: Status of the deletion operation.
    """
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("DELETE_ALL_DOCS: RAGIE_API_KEY environment variable not set")
        return {"status": "error", "message": "RAGIE_API_KEY not set"}
    
    logger.warning("DELETE_ALL_DOCS: Attempting to delete ALL documents from Ragie")
    
    try:
        with Ragie(auth=ragie_api_key) as r_client:
            # Log all available attributes and methods for debugging
            logger.info(f"DELETE_ALL_DOCS: Available client attributes: {dir(r_client)}")
            
            # Check if we can access the documents collection
            if not hasattr(r_client, 'documents'):
                logger.error("DELETE_ALL_DOCS: No documents attribute in Ragie client")
                return {"status": "error", "message": "No documents attribute in Ragie client"}
            
            logger.info(f"DELETE_ALL_DOCS: Available document methods: {dir(r_client.documents)}")
            
            # Try to get all documents, handling pagination
            try:
                all_document_ids = []
                page = 1
                page_size = 100 # Fetch more per page to reduce API calls
                
                if not hasattr(r_client.documents, 'list'):
                    logger.error("DELETE_ALL_DOCS: No list method found in documents")
                    return {"status": "error", "message": "No list method available in Ragie API"}

                logger.info(f"DELETE_ALL_DOCS: Starting document listing with page_size={page_size}")
                
                while True:
                    logger.info(f"DELETE_ALL_DOCS: Fetching page {page}...")
                    # Assuming the list method accepts page and page_size parameters
                    # Adjust parameter names if needed based on Ragie SDK documentation
                    try:
                        documents_response = r_client.documents.list(page=page, page_size=page_size)
                    except TypeError:
                         # Fallback if page/page_size aren't direct args (might be in a request object)
                         # This is a guess; consult Ragie SDK docs for the correct way
                         logger.warning("DELETE_ALL_DOCS: list() might not accept page/page_size directly. Trying request object.")
                         list_request = {"page": page, "page_size": page_size}
                         documents_response = r_client.documents.list(request=list_request)
                         
                    if not documents_response or not hasattr(documents_response, 'documents') or not documents_response.documents:
                        logger.info(f"DELETE_ALL_DOCS: No more documents found on page {page} or later.")
                        break # Exit loop if no documents are returned

                    page_docs = documents_response.documents
                    logger.info(f"DELETE_ALL_DOCS: Found {len(page_docs)} documents on page {page}")
                    
                    for doc in page_docs:
                        doc_id = getattr(doc, 'id', None)
                        if doc_id:
                            all_document_ids.append(doc_id)
                        else:
                            logger.warning("DELETE_ALL_DOCS: Document found without an ID on page {page}, skipping.")
                            
                    # Check if this was the last page (e.g., if count < page_size)
                    # The Ragie API might also return pagination info (next_page_token, total_pages, etc.)
                    # which would be more reliable. Add checks here if available.
                    if len(page_docs) < page_size:
                         logger.info("DELETE_ALL_DOCS: Last page reached (document count less than page size).")
                         break

                    page += 1 # Go to the next page

                logger.info(f"DELETE_ALL_DOCS: Total documents found across all pages: {len(all_document_ids)}")

                if not all_document_ids:
                    logger.info("DELETE_ALL_DOCS: No documents found in total.")
                    return {"status": "success", "message": "No documents found", "count": 0}

                # Now delete all collected document IDs
                deleted_count = 0
                failed_count = 0
                
                if not hasattr(r_client.documents, 'delete'):
                     logger.error("DELETE_ALL_DOCS: No delete method found in documents")
                     return {"status": "error", "message": "No delete method available in Ragie API"}

                for doc_id in all_document_ids:
                    try:
                        logger.info(f"DELETE_ALL_DOCS: Attempting to delete document {doc_id}")
                        r_client.documents.delete(id=doc_id)
                        deleted_count += 1
                        logger.info(f"DELETE_ALL_DOCS: Successfully deleted document {doc_id}")
                    except Exception as del_e:
                        logger.error(f"DELETE_ALL_DOCS: Failed to delete document {doc_id}: {str(del_e)}")
                        failed_count += 1
                
                return {
                    "status": "success",
                    "message": f"Attempted deletion. Deleted: {deleted_count}, Failed: {failed_count} out of {len(all_document_ids)} total documents found.",
                    "deleted_count": deleted_count,
                    "failed_count": failed_count,
                    "total_found": len(all_document_ids)
                }

            except Exception as e:
                logger.error(f"DELETE_ALL_DOCS: Error processing documents with pagination: {str(e)}")
                import traceback
                logger.error(f"DELETE_ALL_DOCS: Traceback: {traceback.format_exc()}")
                return {"status": "error", "message": f"Error processing documents: {str(e)}"}
    
    except Exception as e:
        logger.error(f"DELETE_ALL_DOCS: Error deleting all documents: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}
