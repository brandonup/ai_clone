#!/usr/bin/env python3
"""
Direct test of Ragie API for document retrieval
"""
import os
import sys
import json
from dotenv import load_dotenv
from ragie import Ragie

# Load environment variables
load_dotenv()

def main():
    """Test Ragie API directly for document retrieval."""
    print("=== Testing Ragie API Directly ===")
    
    # Get Ragie API key
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        print("ERROR: RAGIE_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"Using Ragie API key: {ragie_api_key[:4]}...")
    
    # Test document ingestion
    test_content = "This is a test document for direct Ragie API testing."
    test_filename = "direct_test_document.txt"
    
    print(f"Ingesting test document: {test_filename}")
    
    try:
        with Ragie(auth=ragie_api_key) as r_client:
            # Print available client attributes and methods
            print(f"Available client attributes: {dir(r_client)}")
            
            if hasattr(r_client, 'documents'):
                print(f"Available document methods: {dir(r_client.documents)}")
            
            # Ingest test document
            request_data = {
                "file": {
                    "file_name": test_filename,
                    "content": test_content.encode('utf-8'),
                },
                "metadata": {
                    "title": test_filename,
                    "scope": "direct_test"
                }
            }
            
            print(f"Ingesting document with request: {json.dumps({k: v for k, v in request_data.items() if k != 'file'})}")
            
            response = r_client.documents.create(request=request_data)
            
            # Print response
            print(f"Ingestion response type: {type(response)}")
            print(f"Ingestion response attributes: {dir(response)}")
            
            if hasattr(response, 'id'):
                document_id = response.id
                print(f"Successfully ingested document with ID: {document_id}")
            else:
                print(f"Document ingested but no ID found in response")
                document_id = None
            
            # List all documents
            print("\n=== Listing All Documents ===")
            try:
                if hasattr(r_client.documents, 'list'):
                    print("Using documents.list() method")
                    documents_response = r_client.documents.list()
                    
                    if hasattr(documents_response, 'documents'):
                        documents = documents_response.documents
                        print(f"Found {len(documents)} total documents")
                        
                        # Print details of each document
                        for i, doc in enumerate(documents):
                            print(f"\nDocument {i+1}:")
                            print(f"  ID: {getattr(doc, 'id', 'Unknown')}")
                            
                            # Try to access metadata
                            if hasattr(doc, 'metadata'):
                                print(f"  Metadata: {doc.metadata}")
                            else:
                                print(f"  No metadata attribute found")
                                print(f"  Available attributes: {dir(doc)}")
                    else:
                        print("No 'documents' attribute in list response")
                        print(f"Available attributes in response: {dir(documents_response)}")
                else:
                    print("No 'list' method found in documents")
            except Exception as list_e:
                print(f"Error listing documents: {str(list_e)}")
            
            # Test document retrieval
            print("\n=== Testing Document Retrieval ===")
            try:
                # Check available methods for search/retrieval
                if hasattr(r_client, 'retrievals'):
                    print(f"Available retrieval methods: {dir(r_client.retrievals)}")
                    
                    # Try to use retrievals.create method
                    if hasattr(r_client.retrievals, 'create'):
                        print("Using retrievals.create method")
                        
                        retrieval_request = {
                            "query": "test document",
                            "top_k": 5
                        }
                        
                        print(f"Retrieval request: {json.dumps(retrieval_request)}")
                        
                        retrieval_response = r_client.retrievals.create(request=retrieval_request)
                        
                        print(f"Retrieval response type: {type(retrieval_response)}")
                        print(f"Retrieval response attributes: {dir(retrieval_response)}")
                        
                        # Check if response has results
                        if hasattr(retrieval_response, 'results'):
                            results = retrieval_response.results
                            print(f"Found {len(results)} results")
                            
                            # Print details of each result
                            for i, result in enumerate(results):
                                print(f"\nResult {i+1}:")
                                print(f"  ID: {getattr(result, 'id', 'Unknown')}")
                                print(f"  Score: {getattr(result, 'score', 'Unknown')}")
                                
                                # Try to access text content
                                if hasattr(result, 'text'):
                                    print(f"  Text: {result.text[:100]}...")
                                else:
                                    print(f"  No text attribute found")
                                    print(f"  Available attributes: {dir(result)}")
                        else:
                            print("No 'results' attribute in retrieval response")
                            print(f"Available attributes in response: {dir(retrieval_response)}")
                    else:
                        print("No 'create' method found in retrievals")
                else:
                    print("No 'retrievals' attribute found in client")
                    
                # Try to use documents.search method (alternative approach)
                if hasattr(r_client.documents, 'search'):
                    print("\nUsing documents.search method")
                    
                    search_request = {
                        "query": "test document",
                        "top_k": 5
                    }
                    
                    print(f"Search request: {json.dumps(search_request)}")
                    
                    search_response = r_client.documents.search(request=search_request)
                    
                    print(f"Search response type: {type(search_response)}")
                    print(f"Search response attributes: {dir(search_response)}")
                    
                    # Check if response has results
                    if hasattr(search_response, 'results'):
                        results = search_response.results
                        print(f"Found {len(results)} results")
                        
                        # Print details of each result
                        for i, result in enumerate(results):
                            print(f"\nResult {i+1}:")
                            print(f"  ID: {getattr(result, 'id', 'Unknown')}")
                            print(f"  Score: {getattr(result, 'score', 'Unknown')}")
                            
                            # Try to access text content
                            if hasattr(result, 'text'):
                                print(f"  Text: {result.text[:100]}...")
                            else:
                                print(f"  No text attribute found")
                                print(f"  Available attributes: {dir(result)}")
                    else:
                        print("No 'results' attribute in search response")
                        print(f"Available attributes in response: {dir(search_response)}")
                else:
                    print("No 'search' method found in documents")
            except Exception as retrieval_e:
                print(f"Error during retrieval: {str(retrieval_e)}")
            
            # Clean up - delete the test document if we have its ID
            if document_id:
                print(f"\n=== Cleaning Up - Deleting Test Document ===")
                try:
                    if hasattr(r_client.documents, 'delete'):
                        print(f"Deleting document with ID: {document_id}")
                        r_client.documents.delete(id=document_id)
                        print(f"Successfully deleted document")
                    else:
                        print("No 'delete' method found in documents")
                except Exception as del_e:
                    print(f"Error deleting document: {str(del_e)}")
    
    except Exception as e:
        print(f"Error during Ragie API testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
