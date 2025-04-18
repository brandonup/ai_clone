#!/usr/bin/env python3
"""
RAG Diagnostics Tool

This script diagnoses issues with the Retrieval-Augmented Generation (RAG) system,
specifically focusing on API calls to Ragie.ai and other LLM providers.
"""
import os
import sys
import json
import logging
import requests
from dotenv import load_dotenv
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_diagnostics.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("rag_diagnostics")

# Load environment variables
load_dotenv()

# Import project modules
sys.path.append('backend')
try:
    from utils.ragie_utils import query_ragie, ingest_document
    from utils.adaptive_router import run_adaptive_rag
    from utils.clone_manager import load_clones, get_clone_by_id
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def get_clone_data_from_firestore():
    """Get clone data from Firestore."""
    logger.info("Getting clone data from Firestore...")
    
    try:
        # Load all clones from Firestore
        clones = load_clones()
        
        if not clones:
            logger.warning("No clones found in Firestore.")
            return []
        
        logger.info(f"Successfully retrieved {len(clones)} clones from Firestore.")
        return clones
    except Exception as e:
        logger.error(f"Error getting clone data from Firestore: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_random_clone_id():
    """Get a random clone ID from Firestore."""
    clones = get_clone_data_from_firestore()
    if not clones:
        return None
    
    import random
    random_clone = random.choice(clones)
    clone_id = random_clone.get("id")
    logger.info(f"Selected random clone: {random_clone.get('clone_name', 'Unknown')} (ID: {clone_id})")
    return clone_id

def check_api_keys():
    """Check if all required API keys are set and valid."""
    logger.info("Checking API keys...")
    
    api_keys = {
        "RAGIE_API_KEY": os.getenv("RAGIE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY")
    }
    
    missing_keys = [key for key, value in api_keys.items() if not value]
    
    if missing_keys:
        logger.error(f"Missing API keys: {', '.join(missing_keys)}")
    else:
        logger.info("All API keys are set")
    
    # Validate Ragie API key with a simple request
    if api_keys["RAGIE_API_KEY"]:
        try:
            logger.info("Testing Ragie API key...")
            # Create a simple test document to validate the API key
            test_content = f"This is a test document created at {datetime.now().isoformat()}"
            response = ingest_document(
                file_content=test_content,
                filename="test_document.txt",
                clone_name="RAG_Diagnostics",
                collection_name="rag_diagnostics_test",
                delete_after_ingestion=True
            )
            
            if hasattr(response, 'id'):
                logger.info(f"✅ Ragie API key is valid. Document ID: {response.id}")
            else:
                logger.error(f"❌ Ragie API key validation failed. Response: {response}")
        except Exception as e:
            logger.error(f"❌ Ragie API key validation failed with error: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Validate OpenAI API key
    if api_keys["OPENAI_API_KEY"]:
        try:
            logger.info("Testing OpenAI API key...")
            headers = {
                "Authorization": f"Bearer {api_keys['OPENAI_API_KEY']}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "text-embedding-ada-002",
                "input": "This is a test."
            }
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ OpenAI API key is valid")
            else:
                logger.error(f"❌ OpenAI API key validation failed. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
        except Exception as e:
            logger.error(f"❌ OpenAI API key validation failed with error: {str(e)}")
    
    # Validate Cohere API key
    if api_keys["COHERE_API_KEY"]:
        try:
            logger.info("Testing Cohere API key...")
            headers = {
                "Authorization": f"Bearer {api_keys['COHERE_API_KEY']}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "command-light",
                "prompt": "Say hello",
                "max_tokens": 10
            }
            response = requests.post(
                "https://api.cohere.ai/v1/generate",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Cohere API key is valid")
            else:
                logger.error(f"❌ Cohere API key validation failed. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
        except Exception as e:
            logger.error(f"❌ Cohere API key validation failed with error: {str(e)}")
    
    return not missing_keys

def test_ragie_retrieval():
    """Test document retrieval from Ragie.ai."""
    logger.info("Testing Ragie.ai document retrieval...")
    
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Tell me about natural language processing"
    ]
    
    success_count = 0
    
    for query in test_queries:
        try:
            logger.info(f"Testing query: '{query}'")
            documents = query_ragie(query, top_k=3)
            
            if documents:
                logger.info(f"✅ Successfully retrieved {len(documents)} documents for query: '{query}'")
                for i, doc in enumerate(documents):
                    logger.info(f"  Document {i+1} content (first 100 chars): {doc.page_content[:100]}...")
                success_count += 1
            else:
                logger.warning(f"⚠️ No documents retrieved for query: '{query}'")
        except Exception as e:
            logger.error(f"❌ Error retrieving documents for query '{query}': {str(e)}")
            logger.error(traceback.format_exc())
    
    success_rate = success_count / len(test_queries) if test_queries else 0
    logger.info(f"Ragie retrieval success rate: {success_rate * 100:.2f}%")
    
    return success_rate > 0

def test_adaptive_router():
    """Test the adaptive router with different types of queries."""
    logger.info("Testing adaptive router...")
    
    # Get a real clone ID from Firestore
    clone_id = get_random_clone_id()
    if not clone_id:
        logger.warning("No clone IDs available from Firestore. Using 'rag_diagnostics_test' as fallback.")
        clone_id = "rag_diagnostics_test"
        clone_name = "RAG_Diagnostics"
        clone_role = "Assistant"
        persona = "I am a diagnostic assistant"
    else:
        # Get the clone details
        clone = get_clone_by_id(clone_id)
        clone_name = clone.get("clone_name", "RAG_Diagnostics")
        clone_role = clone.get("clone_role", "Assistant")
        persona = clone.get("clone_persona", "I am a diagnostic assistant")
        logger.info(f"Using clone '{clone_name}' (ID: {clone_id}) for testing")
    
    test_cases = [
        {
            "query": "What is artificial intelligence?",
            "expected_route": "vectorstore"
        },
        {
            "query": "What is the latest news about AI?",
            "expected_route": "web_search"
        },
        {
            "query": "Hello, how are you?",
            "expected_route": "base_llm"
        }
    ]
    
    success_count = 0
    
    for case in test_cases:
        query = case["query"]
        expected_route = case["expected_route"]
        
        try:
            logger.info(f"Testing query: '{query}' (expected route: {expected_route})")
            
            # Run the adaptive router with the real clone data
            result = run_adaptive_rag(
                question=query,
                clone_name=clone_name,
                clone_role=clone_role,
                persona=persona
                # Removed vectorstore_name=clone_id as it's not expected
            )
            
            actual_route = result.get("source_path", "unknown")
            logger.info(f"Query: '{query}' was routed to: {actual_route}")
            
            if actual_route == expected_route or (
                expected_route == "vectorstore" and actual_route == "ragie.ai"
            ):
                logger.info(f"✅ Routing successful for query: '{query}'")
                success_count += 1
            else:
                logger.warning(f"⚠️ Unexpected routing for query: '{query}'. Expected: {expected_route}, Got: {actual_route}")
            
            # Check if generation was successful
            generation = result.get("generation", "")
            if generation and generation != "I'm sorry, I encountered an issue processing your request.":
                logger.info(f"✅ Generation successful for query: '{query}'")
                logger.info(f"  Generation (first 100 chars): {generation[:100]}...")
            else:
                logger.warning(f"⚠️ Generation failed or returned error message for query: '{query}'")
                logger.warning(f"  Generation: {generation}")
        
        except Exception as e:
            logger.error(f"❌ Error testing adaptive router for query '{query}': {str(e)}")
            logger.error(traceback.format_exc())
    
    success_rate = success_count / len(test_cases) if test_cases else 0
    logger.info(f"Adaptive router success rate: {success_rate * 100:.2f}%")
    
    return success_rate > 0

def check_rate_limits():
    """Check if any API is hitting rate limits by making multiple requests."""
    logger.info("Checking for rate limit issues...")
    
    # Test Ragie API for rate limits
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if ragie_api_key:
        try:
            logger.info("Testing Ragie API for rate limits...")
            
            # Make 5 quick requests to see if we hit rate limits
            for i in range(5):
                query = f"Test query {i+1}"
                logger.info(f"  Making request {i+1}/5: '{query}'")
                
                documents = query_ragie(query, top_k=1)
                
                if documents:
                    logger.info(f"  ✅ Request {i+1}/5 successful")
                else:
                    logger.warning(f"  ⚠️ Request {i+1}/5 returned no documents")
            
            logger.info("✅ Ragie API rate limit test completed without errors")
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                logger.error(f"❌ Ragie API is hitting rate limits: {str(e)}")
                return False
            else:
                logger.error(f"❌ Error testing Ragie API rate limits: {str(e)}")
    
    # Test OpenAI API for rate limits
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            logger.info("Testing OpenAI API for rate limits...")
            
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Make 5 quick requests to see if we hit rate limits
            for i in range(5):
                logger.info(f"  Making request {i+1}/5")
                
                data = {
                    "model": "text-embedding-ada-002",
                    "input": f"Test input {i+1}"
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info(f"  ✅ Request {i+1}/5 successful")
                elif response.status_code == 429:
                    logger.error(f"  ❌ Request {i+1}/5 hit rate limit (429)")
                    logger.error(f"  Response: {response.text}")
                    return False
                else:
                    logger.warning(f"  ⚠️ Request {i+1}/5 failed with status code: {response.status_code}")
                    logger.warning(f"  Response: {response.text}")
            
            logger.info("✅ OpenAI API rate limit test completed without hitting limits")
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                logger.error(f"❌ OpenAI API is hitting rate limits: {str(e)}")
                return False
            else:
                logger.error(f"❌ Error testing OpenAI API rate limits: {str(e)}")
    
    # Test Cohere API for rate limits
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        try:
            logger.info("Testing Cohere API for rate limits...")
            
            headers = {
                "Authorization": f"Bearer {cohere_api_key}",
                "Content-Type": "application/json"
            }
            
            # Make 5 quick requests to see if we hit rate limits
            for i in range(5):
                logger.info(f"  Making request {i+1}/5")
                
                data = {
                    "model": "command-light",
                    "prompt": f"Test prompt {i+1}",
                    "max_tokens": 5
                }
                
                response = requests.post(
                    "https://api.cohere.ai/v1/generate",
                    headers=headers,
                    json=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info(f"  ✅ Request {i+1}/5 successful")
                elif response.status_code == 429:
                    logger.error(f"  ❌ Request {i+1}/5 hit rate limit (429)")
                    logger.error(f"  Response: {response.text}")
                    return False
                else:
                    logger.warning(f"  ⚠️ Request {i+1}/5 failed with status code: {response.status_code}")
                    logger.warning(f"  Response: {response.text}")
            
            logger.info("✅ Cohere API rate limit test completed without hitting limits")
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                logger.error(f"❌ Cohere API is hitting rate limits: {str(e)}")
                return False
            else:
                logger.error(f"❌ Error testing Cohere API rate limits: {str(e)}")
    
    return True

def main():
    """Run all diagnostic tests and report results."""
    logger.info("=" * 50)
    logger.info("Starting RAG Diagnostics")
    logger.info("=" * 50)
    
    results = {
        "api_keys_valid": False,
        "ragie_retrieval_working": False,
        "adaptive_router_working": False,
        "no_rate_limit_issues": False
    }
    
    # Check API keys
    results["api_keys_valid"] = check_api_keys()
    
    # Test Ragie retrieval
    results["ragie_retrieval_working"] = test_ragie_retrieval()
    
    # Test adaptive router
    results["adaptive_router_working"] = test_adaptive_router()
    
    # Check for rate limit issues
    results["no_rate_limit_issues"] = check_rate_limits()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("RAG Diagnostics Summary")
    logger.info("=" * 50)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test}")
    
    # Provide recommendations based on results
    logger.info("=" * 50)
    logger.info("Recommendations")
    logger.info("=" * 50)
    
    if not results["api_keys_valid"]:
        logger.info("- Check your API keys in the .env file and ensure they are valid")
    
    if not results["ragie_retrieval_working"]:
        logger.info("- Check the Ragie.ai service status")
        logger.info("- Verify your Ragie API key and collection configuration")
        logger.info("- Check if documents were properly ingested into Ragie")
    
    if not results["adaptive_router_working"]:
        logger.info("- Review the adaptive_router.py implementation")
        logger.info("- Check if the router is correctly configured to use Ragie")
    
    if not results["no_rate_limit_issues"]:
        logger.info("- One or more APIs are hitting rate limits")
        logger.info("- Consider implementing rate limiting or backoff strategies")
        logger.info("- Check your API usage and consider upgrading your plan if necessary")
    
    if all(results.values()):
        logger.info("All tests passed! If you're still experiencing issues, check:")
        logger.info("- Network connectivity")
        logger.info("- Firewall or proxy settings")
        logger.info("- Application logs for more specific errors")
    
    logger.info("=" * 50)
    logger.info("End of RAG Diagnostics")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
