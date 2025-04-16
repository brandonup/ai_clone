"""
Alternative implementation for document retrieval with keyword-based fallback
"""
import os
import logging
import json
import re
from ragie import Ragie

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_document(file_content, filename, collection_name=None, delete_after_ingestion=True):
    """
    Ingest a document into Ragie.ai
    
    Args:
        file_content: File content as bytes
        filename: Name of the file
        collection_name: The name of the collection (clone ID) to associate with this document
        delete_after_ingestion: Whether to delete the local copy after ingestion
        
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
        # Sanitize the text
        sanitized_text = sanitize_text(text_content)
        # Convert back to bytes
        processed_content = sanitized_text.encode('utf-8')
    else:
        # Sanitize the text
        sanitized_text = sanitize_text(file_content)
        # Convert to bytes
        processed_content = sanitized_text.encode('utf-8')
    
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
        
        # Add collection_name as clone_id in metadata if provided
        if collection_name:
            metadata["clone_id"] = collection_name
            
        request_data = {
            "file": {
                "file_name": filename,
                "content": processed_content,
            },
            "metadata": metadata
        }
        
        logger.info(f"Ragie ingest request metadata: {request_data['metadata']}")
        
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

def sanitize_text(text):
    """
    Enhanced sanitize_text function to better handle PDF artifacts and improve text quality
    
    Args:
        text: Text to sanitize
        
    Returns:
        str: Sanitized text
    """
    # If text is bytes, decode it
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding bytes: {str(e)}")
            return "[Binary data could not be decoded]"
    
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            logger.error(f"Error converting to string: {str(e)}")
            return "[Data could not be converted to string]"
    
    # Check for binary data or non-printable characters
    # This is a more aggressive check that will catch any text with a high percentage of non-printable characters
    total_chars = len(text)
    if total_chars == 0:
        return ""
        
    printable_chars = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    printable_ratio = printable_chars / total_chars
    
    # If less than 80% of characters are printable, consider it binary data
    if printable_ratio < 0.8:
        logger.warning(f"Detected binary data (printable ratio: {printable_ratio:.2f})")
        return "[Binary data detected and removed]"
    
    # Check for PDF artifacts
    import re
    
    # Expanded list of PDF-specific markers
    pdf_patterns = [
        r'endstream\s*endobj',
        r'\d+\s+\d+\s+obj\s*<<.*?>>',
        r'<<.*?>>',
        r'/Filter.*?/FlateDecode',
        r'/Length\s+\d+',
        r'/Type/.*?/',
        r'stream\s*h',
        r'endstream',
        r'endobj',
        r'\[\d+ \d+ \d+\]',
        r'/W\[\d+ \d+ \d+\]',
        r'/Root \d+ \d+ R',
        r'/Info \d+ \d+ R',
        r'/ID\[.*?\]',
        r'/Size \d+',
        r'/Predictor \d+',
        r'/Columns \d+',
        r'/DecodeParms',
        r'/FontDescriptor',
        r'/FontFile[23]?',
        r'/XObject',
        r'/ExtGState',
        r'/ColorSpace',
        r'/Pattern',
        r'/Shading',
        r'/Font',
        r'/ProcSet',
        r'/MediaBox',
        r'/CropBox',
        r'/Rotate',
        r'/Resources',
        r'/Contents',
        r'/Parent',
        r'/Kids',
        r'/Count',
        r'/Metadata',
        r'/Outlines',
        r'/Pages',
        r'/Page',
        r'/Catalog',
    ]
    
    # Check for PDF markers
    pdf_marker_count = sum(1 for pattern in pdf_patterns if re.search(pattern, text))
    # Increased threshold from 3 to 5 to reduce false positives
    if pdf_marker_count >= 5: 
        logger.warning(f"Detected PDF markers ({pdf_marker_count} markers)")
        return "[PDF content detected and removed]"
    
    # Check for random character patterns that might indicate binary data
    # This pattern looks for text that has a high ratio of special characters to alphanumeric characters
    special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\r\t.,;:!?()[]{}"\'-_')
    alpha_chars = sum(1 for c in text if c.isalnum())
    
    if alpha_chars > 0:
        special_ratio = special_chars / alpha_chars
        if special_ratio > 0.5:  # If more than 50% of non-space characters are special characters
            logger.warning(f"Detected unusual character distribution (special ratio: {special_ratio:.2f})")
            return "[Content with unusual character distribution removed]"
    
    # Check for random character sequences (like the example provided by the user)
    # This pattern looks for text with many short "words" separated by spaces
    words = text.split()
    short_words = sum(1 for word in words if len(word) <= 2)
    if len(words) > 10 and short_words / len(words) > 0.5:
        logger.warning(f"Detected random character sequence (short word ratio: {short_words/len(words):.2f})")
        return "[Random character sequence detected and removed]"
    
    # Fix common OCR/PDF extraction issues
    text = text.replace("ﬂ", "fl")  # Replace ligatures
    text = text.replace("ﬁ", "fi")
    text = text.replace("iaweb", "Viaweb")  # Fix specific known issues
    text = text.replace("V iaweb", "Viaweb")
    
    # Replace any remaining non-printable characters
    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix broken sentences (periods without spaces)
    text = re.sub(r'(\w)\.([A-Z])', r'\1. \2', text)
    
    return text.strip()


def query_ragie(query: str, top_k: int = 5, clone_id: str = None) -> list:
    """
    Query Ragie.ai to retrieve relevant documents.

    Args:
        query (str): The user's query.
        top_k (int): The maximum number of documents to retrieve.
        clone_id (str): Optional clone ID to filter documents by.

    Returns:
        list: A list of LangChain Document objects.
    """
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("RAGIE_API_KEY environment variable not set for querying.")
        return []

    logger.info(f"Querying Ragie.ai with: '{query}' (top_k={top_k})")

    try:
        with Ragie(auth=ragie_api_key) as r_client:
            # Assuming a search method exists, adjust if the actual method is different
            # Create search request with optional clone_id filter
            search_request = {
                "query": query,
                "top_k": top_k
            }
            
            # Add clone_id filter if provided
            if clone_id:
                search_request["filter"] = {"clone_id": clone_id}
                logger.info(f"Filtering search results by clone_id: {clone_id}")
            logger.debug(f"Ragie search request: {search_request}")
            response = r_client.search.create(request=search_request) # Adjust method if needed

            # Process the response (assuming response.results is a list of hits)
            documents = []
            if hasattr(response, 'results') and response.results:
                logger.info(f"Received {len(response.results)} results from Ragie.")
                for result in response.results:
                    # Extract content and metadata (adjust based on actual Ragie response structure)
                    content = getattr(result, 'text', '')
                    metadata = getattr(result, 'metadata', {})
                    score = getattr(result, 'score', None) # Get score if available

                    # Ensure metadata is a dictionary
                    if not isinstance(metadata, dict):
                        metadata = {"source_data": str(metadata)} # Convert non-dict metadata

                    # Add score to metadata if available
                    if score is not None:
                        metadata['score'] = score
                    metadata['source'] = 'ragie.ai' # Indicate the source

                    # Create LangChain Document
                    if content:
                        from langchain_core.documents import Document # Local import
                        documents.append(Document(page_content=content, metadata=metadata))
                    else:
                         logger.warning(f"Ragie result skipped due to empty content. Metadata: {metadata}")

            else:
                logger.info("No results received from Ragie.")

            return documents

    except AttributeError:
        logger.error("The 'ragie' client object might not have a 'search.create' method. Please check the Ragie library documentation for the correct query method.")
        return []
    except Exception as e:
        logger.error(f"Error querying Ragie.ai: {str(e)}")
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
    if not clone_id:
        logger.error("No clone ID provided for document deletion")
        return {"status": "error", "message": "No clone ID provided"}
        
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("RAGIE_API_KEY environment variable not set for document deletion")
        return {"status": "error", "message": "RAGIE_API_KEY not set"}
    
    logger.info(f"Attempting to delete all documents for clone: {clone_id}")
    
    try:
        with Ragie(auth=ragie_api_key) as r_client:
            # First, we need to list all documents that match the clone_id
            # This assumes Ragie has a list/search method for documents
            try:
                # Try to list documents with a filter for the clone_id
                # Note: This is based on inference of the API - adjust if needed
                list_request = {
                    "filter": {"clone_id": clone_id}
                }
                
                # Attempt to list documents
                documents = r_client.documents.list(request=list_request)
                
                if not documents or not hasattr(documents, 'results') or not documents.results:
                    logger.info(f"No documents found for clone {clone_id}")
                    return {"status": "success", "message": "No documents found to delete", "count": 0}
                
                # Track deletion results
                deleted_count = 0
                failed_count = 0
                
                # Delete each document
                for doc in documents.results:
                    doc_id = getattr(doc, 'id', None)
                    if not doc_id:
                        logger.warning(f"Document without ID found for clone {clone_id}, skipping")
                        failed_count += 1
                        continue
                        
                    try:
                        # Delete the document
                        r_client.documents.delete(id=doc_id)
                        deleted_count += 1
                        logger.info(f"Deleted document {doc_id} for clone {clone_id}")
                    except Exception as del_e:
                        logger.error(f"Failed to delete document {doc_id}: {str(del_e)}")
                        failed_count += 1
                
                return {
                    "status": "success",
                    "message": f"Deleted {deleted_count} documents, failed to delete {failed_count}",
                    "deleted_count": deleted_count,
                    "failed_count": failed_count
                }
                
            except AttributeError:
                # If the list method doesn't exist, try an alternative approach
                logger.warning("documents.list method not found, trying alternative approach")
                
                # Try to use search to find documents by clone_id
                search_request = {
                    "query": "",  # Empty query to match all
                    "filter": {"clone_id": clone_id},
                    "top_k": 1000  # Get a large number of results
                }
                
                search_results = r_client.search.create(request=search_request)
                
                if not search_results or not hasattr(search_results, 'results') or not search_results.results:
                    logger.info(f"No documents found for clone {clone_id} using search")
                    return {"status": "success", "message": "No documents found to delete", "count": 0}
                
                # Track deletion results
                deleted_count = 0
                failed_count = 0
                
                # Delete each document found in search
                for result in search_results.results:
                    doc_id = getattr(result, 'document_id', None)
                    if not doc_id:
                        logger.warning(f"Search result without document_id found for clone {clone_id}, skipping")
                        failed_count += 1
                        continue
                        
                    try:
                        # Delete the document
                        r_client.documents.delete(id=doc_id)
                        deleted_count += 1
                        logger.info(f"Deleted document {doc_id} for clone {clone_id}")
                    except Exception as del_e:
                        logger.error(f"Failed to delete document {doc_id}: {str(del_e)}")
                        failed_count += 1
                
                return {
                    "status": "success",
                    "message": f"Deleted {deleted_count} documents, failed to delete {failed_count}",
                    "deleted_count": deleted_count,
                    "failed_count": failed_count
                }
    
    except Exception as e:
        logger.error(f"Error deleting documents for clone {clone_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error: {str(e)}"}
