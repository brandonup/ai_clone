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
        
    Returns:
        dict: Response from Ragie API
    """
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        raise ValueError("RAGIE_API_KEY environment variable not set")
    
    logger.info(f"Ingesting document: {filename}")
    
    # Convert to string if bytes
    if isinstance(file_content, bytes):
        processed_content = file_content
    else:
        # Convert to bytes if string
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
        request_data = {
            "file": {
                "file_name": filename,
                "content": processed_content,
            },
            "metadata": {"title": filename, "scope": "coach_data"}
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
