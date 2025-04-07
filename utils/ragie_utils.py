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

def preprocess_document_content(content):
    """
    Preprocess document content to improve chunking and indexing
    
    Args:
        content: Document content as bytes or string
        
    Returns:
        bytes: Processed content
    """
    # Convert to string if bytes
    if isinstance(content, bytes):
        text = content.decode('utf-8', errors='replace')
    else:
        text = content
    
    # Fix common OCR/PDF extraction issues
    text = text.replace("ﬂ", "fl")  # Replace ligatures
    text = text.replace("ﬁ", "fi")
    text = text.replace("iaweb", "Viaweb")  # Fix specific known issues
    text = text.replace("V iaweb", "Viaweb")
    
    # Normalize line breaks (replace multiple line breaks with single line break)
    text = re.sub(r'\n{2,}', '\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure paragraphs are properly separated
    text = re.sub(r'(\.\s+)([A-Z])', r'\1\n\2', text)
    
    # Convert back to bytes
    return text.encode('utf-8')

def ingest_document(file_content, filename):
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
    
    # Preprocess document content to improve chunking
    processed_content = preprocess_document_content(file_content)
    
    # Store a local copy of the document for keyword-based fallback
    try:
        doc_text = processed_content.decode('utf-8')
        doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents', filename)
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        with open(doc_path, 'w') as f:
            f.write(doc_text)
        logger.info(f"Stored local copy of document at {doc_path}")
    except Exception as e:
        logger.error(f"Error storing local copy of document: {str(e)}")
    
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
            except Exception as e:
                logger.info(f"Ragie ingest response: {str(response)}")
                
            return response
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise

def keyword_search(query, top_n=5):
    """
    Perform a simple keyword-based search on locally stored documents
    
    Args:
        query: User question
        top_n: Number of top chunks to return
        
    Returns:
        list: List of text chunks
    """
    logger.info(f"Performing keyword search for: {query}")
    
    # Directory containing locally stored documents
    doc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents')
    if not os.path.exists(doc_dir):
        logger.warning(f"Document directory not found: {doc_dir}")
        return []
    
    # Extract keywords from query
    keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))
    logger.info(f"Keywords extracted: {keywords}")
    
    # Add entity-specific keywords
    entity_mappings = {
        "viaweb": ["viaweb", "iaweb", "v iaweb"],
        "seed": ["seed", "funding", "investment", "money", "capital"],
    }
    
    expanded_keywords = set(keywords)
    for keyword in keywords:
        if keyword in entity_mappings:
            expanded_keywords.update(entity_mappings[keyword])
    
    # Remove common stop words
    stop_words = {'the', 'and', 'for', 'with', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'this', 'that', 'have', 'has', 'had', 'not', 'are', 'about'}
    expanded_keywords = expanded_keywords - stop_words
    
    logger.info(f"Expanded keywords: {expanded_keywords}")
    
    results = []
    
    # Search through all documents
    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Clean content before searching
            content = sanitize_text(content)
            
            # Split into paragraphs
            paragraphs = re.split(r'\n+', content)
            
            for para in paragraphs:
                if len(para.strip()) < 20:  # Skip very short paragraphs
                    continue
                
                # Count keyword matches
                para_lower = para.lower()
                matches = sum(1 for keyword in expanded_keywords if keyword in para_lower)
                
                if matches > 0:
                    # Score based on keyword density and paragraph length
                    score = matches / (len(para.split()) + 1) * 100
                    results.append((para, score))
        except Exception as e:
            logger.error(f"Error searching document {filename}: {str(e)}")
    
    # Sort by score (highest first) and take top_n
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Log the results
    logger.info(f"Keyword search found {len(results)} matching paragraphs")
    for i, (text, score) in enumerate(results[:top_n]):
        logger.info(f"Match {i+1}: score={score}, text={text[:100]}...")
    
    # Return just the text of the top results, ensuring they are properly sanitized
    return [sanitize_text(text) for text, _ in results[:top_n]]

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

def retrieve_chunks(query, top_n=2):
    """
    Retrieve relevant chunks from Ragie.ai for a given query
    with keyword-based fallback and optimized for token limits
    
    Args:
        query: User question
        top_n: Number of top chunks to return
        
    Returns:
        list: List of text chunks
    """
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        raise ValueError("RAGIE_API_KEY environment variable not set")
    
    logger.info(f"Original query: {query}")
    
    # Normalize entity names in the query
    normalized_query = query.replace("iaweb", "Viaweb").replace("V iaweb", "Viaweb")
    if normalized_query != query:
        logger.info(f"Normalized query: {normalized_query}")
        query = normalized_query
    
    # Try multiple query variations to improve retrieval
    queries = [
        query,  # Original query
        f"Find information about: {query}",  # Directive format
        query.replace("?", "").strip(),  # Remove question mark
        " ".join(query.lower().split()),  # Normalize spacing and case
    ]
    
    # Add entity-specific query variations
    if "viaweb" in query.lower() or "iaweb" in query.lower():
        queries.append(f"Information about Viaweb {query}")
        queries.append(f"Viaweb history {query}")
        
    if "seed" in query.lower() and ("money" in query.lower() or "funding" in query.lower()):
        queries.append(f"Viaweb seed funding amount")
        queries.append(f"How much seed money did Viaweb receive")
    
    all_chunks = []
    
    with Ragie(auth=ragie_api_key) as r_client:
        # First, log all available documents to verify content is indexed
        try:
            docs_response = r_client.documents.list()
            # Safely log response
            try:
                if isinstance(docs_response, dict):
                    logger.info(f"Available documents in Ragie: {json.dumps(docs_response, indent=2)}")
                else:
                    logger.info(f"Available documents in Ragie (non-JSON format): {str(docs_response)}")
            except Exception as e:
                logger.error(f"Error logging documents list: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
        
        # Try each query variation
        for q in queries:
            if not q.strip():  # Skip empty queries
                continue
                
            logger.info(f"Trying query variation: {q}")
            
            request_data = {
                "query": q,
                "rerank": True,
                "limit": 20,  # Reduced from 30 to 20 to focus on more relevant matches
                "min_score": 0.15  # Lowered from 0.2 to 0.15 to be more inclusive
            }
            
            logger.info(f"Ragie retrieve request: {json.dumps(request_data, indent=2)}")
            
            try:
                result = r_client.retrievals.retrieve(request=request_data)
                
                # Safely log response
                try:
                    if isinstance(result, dict):
                        logger.info(f"Ragie retrieve full response: {json.dumps(result, indent=2)}")
                    else:
                        logger.info(f"Ragie retrieve response (non-JSON format): {str(result)}")
                except Exception as e:
                    logger.error(f"Error logging retrieve response: {str(e)}")
                
                # Extract chunks with scores
                if result and hasattr(result, 'scored_chunks'):
                    # Handle object-style response (newer Ragie SDK)
                    chunks_list = result.scored_chunks
                    logger.info(f"Query '{q}' returned {len(chunks_list)} chunks")
                    for chunk in chunks_list:
                        if hasattr(chunk, 'score') and hasattr(chunk, 'text'):
                            logger.info(f"Chunk score: {chunk.score}, text: {chunk.text[:100]}...")
                            all_chunks.append((chunk.text, chunk.score))
                elif result and isinstance(result, dict) and "scored_chunks" in result:
                    # Handle dictionary-style response (older Ragie SDK)
                    logger.info(f"Query '{q}' returned {len(result['scored_chunks'])} chunks")
                    for chunk in result["scored_chunks"]:
                        if "score" in chunk and "text" in chunk:
                            logger.info(f"Chunk score: {chunk['score']}, text: {chunk['text'][:100]}...")
                            all_chunks.append((chunk["text"], chunk["score"]))
                else:
                    logger.info(f"Query '{q}' returned no chunks or unexpected response format")
            except Exception as e:
                logger.error(f"Error retrieving chunks for query '{q}': {str(e)}")
    
    # If Ragie didn't return any useful chunks, try keyword-based search
    if not all_chunks:
        logger.info("No chunks returned from Ragie, falling back to keyword search")
        keyword_results = keyword_search(query, top_n)
        if keyword_results:
            logger.info("Keyword search found results, using these instead")
            return keyword_results
    
    # Remove duplicates while preserving order
    unique_chunks = []
    seen_texts = set()
    
    for text, score in all_chunks:
        # Clean text before deduplication
        cleaned_text = sanitize_text(text)
        if cleaned_text not in seen_texts:
            seen_texts.add(cleaned_text)
            unique_chunks.append((cleaned_text, score))
    
    # Sort by score (highest first) and take top_n
    unique_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Use tiktoken for accurate token counting if available
    try:
        import tiktoken
        def estimate_tokens(text):
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(enc.encode(text))
        logger.info("Using tiktoken for accurate token counting")
    except ImportError:
        # Fall back to character-based estimation
        def estimate_tokens(text):
            return len(text) // 4
        logger.info("Tiktoken not available, using character-based token estimation")
    
    # Calculate max tokens per chunk to stay within limits
    # Assuming context window of ~4000 tokens, reserve ~1000 for question and answer
    max_total_tokens = 3000
    max_tokens_per_chunk = max_total_tokens // top_n
    
    # Limit chunk size to avoid token limit issues
    limited_chunks = []
    for text, score in unique_chunks[:top_n]:
        # Truncate long chunks based on estimated token count
        token_count = estimate_tokens(text)
        if token_count > max_tokens_per_chunk:
            # Try to truncate at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            truncated_text = ""
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = estimate_tokens(sentence)
                if current_tokens + sentence_tokens <= max_tokens_per_chunk:
                    truncated_text += sentence + " "
                    current_tokens += sentence_tokens
                else:
                    break
            
            if truncated_text:
                text = truncated_text.strip() + "..."
            else:
                # If sentence-based truncation didn't work, fall back to word-based
                words = text.split()
                estimated_word_limit = int(max_tokens_per_chunk * 0.75)  # Estimate words per token
                text = " ".join(words[:estimated_word_limit]) + "..."
        
        limited_chunks.append((text, score))
    
    # Log the final selected chunks
    logger.info(f"Selected {min(top_n, len(limited_chunks))} chunks from {len(unique_chunks)} unique chunks")
    for i, (text, score) in enumerate(limited_chunks):
        token_estimate = estimate_tokens(text)
        logger.info(f"Chunk {i+1}: ~{token_estimate} tokens, score={score}, text={text[:100]}...")
    
    # Return just the text of the top results
    return [text for text, _ in limited_chunks]
