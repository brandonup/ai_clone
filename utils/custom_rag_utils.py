"""
Custom RAG implementation with Small-to-Big chunking and sliding window
using Qdrant for vector storage and retrieval
"""
import os
import logging
import json
import re
import uuid
from typing import List, Dict, Tuple, Any, Optional, Union

import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LARGE_CHUNK_SIZE = 512  # tokens
SMALL_CHUNK_SIZE = 175  # tokens
LARGE_CHUNK_OVERLAP = 100  # tokens
SMALL_CHUNK_OVERLAP = 50  # tokens
COLLECTION_NAME = "ai_coach_docs"
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension

class DocumentProcessor:
    """
    Process documents with Small-to-Big chunking and sliding window
    """
    
    def __init__(self, 
                 large_chunk_size: int = LARGE_CHUNK_SIZE,
                 small_chunk_size: int = SMALL_CHUNK_SIZE,
                 large_chunk_overlap: int = LARGE_CHUNK_OVERLAP,
                 small_chunk_overlap: int = SMALL_CHUNK_OVERLAP):
        """
        Initialize the document processor
        
        Args:
            large_chunk_size: Size of large chunks in tokens
            small_chunk_size: Size of small chunks in tokens
            large_chunk_overlap: Overlap between large chunks in tokens
            small_chunk_overlap: Overlap between small chunks in tokens
        """
        self.large_chunk_size = large_chunk_size
        self.small_chunk_size = small_chunk_size
        self.large_chunk_overlap = large_chunk_overlap
        self.small_chunk_overlap = small_chunk_overlap
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            logger.info("Using tiktoken for token counting")
        except:
            logger.warning("Tiktoken not available, using character-based token estimation")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to character-based estimation (1 token ≈ 4 chars)
            return len(text) // 4
    
    def preprocess_document_content(self, content: Union[str, bytes]) -> str:
        """
        Preprocess document content to improve chunking and indexing
        
        Args:
            content: Document content as bytes or string
            
        Returns:
            str: Processed content
        """
        # Convert to string if bytes
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='replace')
        else:
            text = content
        
        # Fix common OCR/PDF extraction issues
        text = text.replace("ﬂ", "fl")  # Replace ligatures
        text = text.replace("ﬁ", "fi")
        
        # Normalize line breaks (replace multiple line breaks with single line break)
        text = re.sub(r'\n{2,}', '\n', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure paragraphs are properly separated
        text = re.sub(r'(\.\s+)([A-Z])', r'\1\n\2', text)
        
        return text
    
    def create_chunks_with_sliding_window(self, 
                                          text: str, 
                                          chunk_size: int, 
                                          chunk_overlap: int) -> List[str]:
        """
        Create chunks with sliding window
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List[str]: List of chunks
        """
        if not text:
            return []
        
        # Split text into tokens
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            
            # Create chunks with sliding window
            chunks = []
            i = 0
            while i < len(tokens):
                # Get chunk tokens
                chunk_end = min(i + chunk_size, len(tokens))
                chunk_tokens = tokens[i:chunk_end]
                
                # Convert back to text
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                # Slide window
                i += chunk_size - chunk_overlap
                
                # Break if we've reached the end
                if i >= len(tokens):
                    break
            
            return chunks
        else:
            # Fallback to character-based chunking
            # Estimate characters per token (4 chars ≈ 1 token)
            char_chunk_size = chunk_size * 4
            char_chunk_overlap = chunk_overlap * 4
            
            # Create chunks with sliding window
            chunks = []
            i = 0
            while i < len(text):
                # Get chunk
                chunk_end = min(i + char_chunk_size, len(text))
                chunk = text[i:chunk_end]
                chunks.append(chunk)
                
                # Slide window
                i += char_chunk_size - char_chunk_overlap
                
                # Break if we've reached the end
                if i >= len(text):
                    break
            
            return chunks
    
    def process_document(self, 
                         content: Union[str, bytes], 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a document with Small-to-Big chunking and sliding window
        
        Args:
            content: Document content as bytes or string
            metadata: Document metadata
            
        Returns:
            Dict: Processed document with chunks
        """
        if metadata is None:
            metadata = {}
        
        # Preprocess document content
        text = self.preprocess_document_content(content)
        
        # Create large chunks with sliding window
        large_chunks = self.create_chunks_with_sliding_window(
            text, 
            self.large_chunk_size, 
            self.large_chunk_overlap
        )
        
        # Process large chunks
        processed_large_chunks = []
        all_small_chunks = []
        
        for i, large_chunk in enumerate(large_chunks):
            # Create large chunk ID
            large_chunk_id = f"{metadata.get('doc_id', 'doc')}_{i}"
            
            # Create large chunk metadata
            large_chunk_metadata = {
                "chunk_id": large_chunk_id,
                "doc_id": metadata.get("doc_id", ""),
                "doc_title": metadata.get("title", ""),
                "chunk_type": "large",
                "chunk_index": i,
            }
            
            # Add large chunk
            processed_large_chunks.append({
                "text": large_chunk,
                "metadata": large_chunk_metadata
            })
            
            # Create small chunks with sliding window
            small_chunks = self.create_chunks_with_sliding_window(
                large_chunk, 
                self.small_chunk_size, 
                self.small_chunk_overlap
            )
            
            # Process small chunks
            for j, small_chunk in enumerate(small_chunks):
                # Create small chunk ID
                small_chunk_id = f"{large_chunk_id}_{j}"
                
                # Create small chunk metadata
                small_chunk_metadata = {
                    "chunk_id": small_chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "doc_title": metadata.get("title", ""),
                    "chunk_type": "small",
                    "chunk_index": j,
                    "parent_id": large_chunk_id,
                    "parent_text": large_chunk,
                }
                
                # Add small chunk
                all_small_chunks.append({
                    "text": small_chunk,
                    "metadata": small_chunk_metadata
                })
        
        # Return processed document
        return {
            "doc_id": metadata.get("doc_id", ""),
            "title": metadata.get("title", ""),
            "large_chunks": processed_large_chunks,
            "small_chunks": all_small_chunks,
        }

class QdrantStore:
    """
    Store and retrieve document chunks using Qdrant
    """
    
    def __init__(self, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 collection_name: str = COLLECTION_NAME):
        """
        Initialize the Qdrant store
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Qdrant collection name
        """
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            # Use in-memory Qdrant for testing
            self.client = QdrantClient(":memory:")
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Ensure the Qdrant collection exists
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embeddings
        """
        # Generate embeddings
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        
        return embeddings
    
    def store_document(self, processed_doc: Dict[str, Any]) -> None:
        """
        Store a processed document in Qdrant
        
        Args:
            processed_doc: Processed document with chunks
        """
        # Extract small chunks
        small_chunks = processed_doc["small_chunks"]
        
        if not small_chunks:
            logger.warning(f"No small chunks to store for document: {processed_doc['doc_id']}")
            return
        
        # Extract texts and metadata
        texts = [chunk["text"] for chunk in small_chunks]
        metadatas = [chunk["metadata"] for chunk in small_chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create points
        points = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            # Create point ID
            point_id = str(uuid.uuid4())
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text,
                    "chunk_id": metadata["chunk_id"],
                    "doc_id": metadata["doc_id"],
                    "doc_title": metadata["doc_title"],
                    "chunk_type": metadata["chunk_type"],
                    "chunk_index": metadata["chunk_index"],
                    "parent_id": metadata["parent_id"],
                    "parent_text": metadata["parent_text"],
                }
            )
            
            points.append(point)
        
        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Stored {len(points)} small chunks for document: {processed_doc['doc_id']}")
    
    def search(self, 
               query: str, 
               limit: int = 5, 
               filter: Optional[Filter] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query: Query text
            limit: Maximum number of results
            filter: Qdrant filter
            
        Returns:
            List[Dict]: List of search results
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search for similar chunks
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter
        )
        
        # Extract results
        results = []
        for scored_point in search_result:
            # Extract point
            point_id = scored_point.id
            score = scored_point.score
            payload = scored_point.payload
            
            # Create result
            result = {
                "id": point_id,
                "score": score,
                "text": payload["text"],
                "chunk_id": payload["chunk_id"],
                "doc_id": payload["doc_id"],
                "doc_title": payload["doc_title"],
                "chunk_type": payload["chunk_type"],
                "chunk_index": payload["chunk_index"],
                "parent_id": payload["parent_id"],
                "parent_text": payload["parent_text"],
            }
            
            results.append(result)
        
        return results

def ingest_document(file_content: Union[str, bytes], filename: str) -> Dict[str, Any]:
    """
    Ingest a document into the custom RAG system
    
    Args:
        file_content: File content as bytes or string
        filename: Name of the file
        
    Returns:
        Dict: Response with document ID
    """
    # Create document ID
    doc_id = str(uuid.uuid4())
    
    # Create document metadata
    metadata = {
        "doc_id": doc_id,
        "title": filename,
    }
    
    # Process document
    processor = DocumentProcessor()
    processed_doc = processor.process_document(file_content, metadata)
    
    # Store document
    store = QdrantStore()
    store.store_document(processed_doc)
    
    # Store a local copy of the document for fallback
    try:
        if isinstance(file_content, bytes):
            doc_text = file_content.decode('utf-8', errors='replace')
        else:
            doc_text = file_content
            
        doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'documents', filename)
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        
        with open(doc_path, 'w') as f:
            f.write(doc_text)
            
        logger.info(f"Stored local copy of document at {doc_path}")
    except Exception as e:
        logger.error(f"Error storing local copy of document: {str(e)}")
    
    # Return response
    return {
        "status": "success",
        "doc_id": doc_id,
    }

def retrieve_chunks(query: str, top_n: int = 2) -> List[str]:
    """
    Retrieve relevant chunks for a given query
    
    Args:
        query: User question
        top_n: Number of top chunks to return
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info(f"Original query: {query}")
    
    # Try multiple query variations to improve retrieval
    queries = [
        query,  # Original query
        f"Find information about: {query}",  # Directive format
        query.replace("?", "").strip(),  # Remove question mark
        " ".join(query.lower().split()),  # Normalize spacing and case
    ]
    
    # Initialize Qdrant store
    store = QdrantStore()
    
    # Search for each query variation
    all_results = []
    
    for q in queries:
        if not q.strip():  # Skip empty queries
            continue
            
        logger.info(f"Trying query variation: {q}")
        
        # Search for similar chunks
        results = store.search(q, limit=10)
        
        if results:
            logger.info(f"Query '{q}' returned {len(results)} chunks")
            all_results.extend(results)
        else:
            logger.info(f"Query '{q}' returned no chunks")
    
    # If no results, try keyword-based search as fallback
    if not all_results:
        logger.info("No chunks returned from vector search, falling back to keyword search")
        
        # Import keyword_search from ragie_utils
        from utils.ragie_utils import keyword_search
        
        keyword_results = keyword_search(query, top_n)
        
        if keyword_results:
            logger.info("Keyword search found results, using these instead")
            return keyword_results
    
    # Remove duplicates while preserving order
    unique_results = []
    seen_parent_ids = set()
    
    for result in all_results:
        parent_id = result["parent_id"]
        
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            unique_results.append(result)
    
    # Sort by score (highest first)
    unique_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Get parent chunks for top results
    parent_chunks = []
    
    for result in unique_results[:top_n]:
        parent_text = result["parent_text"]
        parent_chunks.append(parent_text)
    
    # Log the final selected chunks
    logger.info(f"Selected {len(parent_chunks)} parent chunks")
    for i, text in enumerate(parent_chunks):
        logger.info(f"Parent chunk {i+1}: {text[:100]}...")
    
    return parent_chunks
