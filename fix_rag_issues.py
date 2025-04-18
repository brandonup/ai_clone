#!/usr/bin/env python3
"""
RAG Issues Fixer

This script attempts to fix common issues with the Retrieval-Augmented Generation (RAG) system
based on the results of the diagnostic tool.
"""
import os
import sys
import json
import logging
import requests
import argparse
import traceback
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_fixes.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("rag_fixes")

# Load environment variables
load_dotenv()

# Import project modules
sys.path.append('backend')
try:
    from utils.ragie_utils import query_ragie, ingest_document, delete_all_documents_ragie
    from utils.adaptive_router import run_adaptive_rag
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def backup_env_file():
    """Create a backup of the .env file."""
    try:
        env_path = os.path.join('backend', '.env')
        backup_path = os.path.join('backend', f'.env.backup.{datetime.now().strftime("%Y%m%d%H%M%S")}')
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup of .env file at {backup_path}")
            return True
        else:
            logger.error(f"Could not find .env file at {env_path}")
            return False
    except Exception as e:
        logger.error(f"Error backing up .env file: {e}")
        return False

def validate_api_key(key_name, test_function):
    """Validate an API key using the provided test function."""
    api_key = os.getenv(key_name)
    if not api_key:
        logger.error(f"{key_name} is not set in the .env file")
        return False
    
    try:
        logger.info(f"Testing {key_name}...")
        is_valid = test_function(api_key)
        if is_valid:
            logger.info(f"✅ {key_name} is valid")
            return True
        else:
            logger.error(f"❌ {key_name} is invalid")
            return False
    except Exception as e:
        logger.error(f"❌ Error validating {key_name}: {e}")
        return False

def test_ragie_api_key(api_key):
    """Test if a Ragie API key is valid."""
    try:
        # Create a simple test document to validate the API key
        test_content = f"This is a test document created at {datetime.now().isoformat()}"
        
        # Save the current API key
        current_key = os.environ.get("RAGIE_API_KEY")
        
        # Set the API key for testing
        os.environ["RAGIE_API_KEY"] = api_key
        
        # Test the API key
        response = ingest_document(
            file_content=test_content,
            filename="test_document.txt",
            clone_name="RAG_Fixer",
            collection_name="rag_fixer_test",
            delete_after_ingestion=True
        )
        
        # Restore the original API key
        if current_key:
            os.environ["RAGIE_API_KEY"] = current_key
        else:
            os.environ.pop("RAGIE_API_KEY", None)
        
        # Check if the response has an ID (indicating success)
        return hasattr(response, 'id')
    except Exception as e:
        logger.error(f"Error testing Ragie API key: {e}")
        
        # Restore the original API key
        if current_key:
            os.environ["RAGIE_API_KEY"] = current_key
        else:
            os.environ.pop("RAGIE_API_KEY", None)
        
        return False

def test_openai_api_key(api_key):
    """Test if an OpenAI API key is valid."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
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
        
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing OpenAI API key: {e}")
        return False

def test_cohere_api_key(api_key):
    """Test if a Cohere API key is valid."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
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
        
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing Cohere API key: {e}")
        return False

def test_serper_api_key(api_key):
    """Test if a Serper API key is valid."""
    try:
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "q": "test query"
        }
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=data,
            timeout=10
        )
        
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing Serper API key: {e}")
        return False

def update_api_key(key_name, new_key):
    """Update an API key in the .env file."""
    try:
        env_path = os.path.join('backend', '.env')
        
        # Read the current .env file
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Check if the key already exists
        key_exists = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key_name}="):
                lines[i] = f"{key_name}={new_key}\n"
                key_exists = True
                break
        
        # If the key doesn't exist, add it
        if not key_exists:
            lines.append(f"{key_name}={new_key}\n")
        
        # Write the updated .env file
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        logger.info(f"Updated {key_name} in .env file")
        return True
    except Exception as e:
        logger.error(f"Error updating {key_name} in .env file: {e}")
        return False

def fix_api_keys():
    """Fix API key issues."""
    logger.info("Fixing API key issues...")
    
    # Backup the .env file
    if not backup_env_file():
        logger.error("Failed to backup .env file. Aborting API key fixes.")
        return False
    
    # Check and fix Ragie API key
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if ragie_api_key and not validate_api_key("RAGIE_API_KEY", test_ragie_api_key):
        logger.warning("Ragie API key is invalid. Please provide a new key.")
        new_key = input("Enter new Ragie API key (or press Enter to skip): ").strip()
        if new_key:
            if update_api_key("RAGIE_API_KEY", new_key):
                # Reload environment variables
                load_dotenv()
                logger.info("Ragie API key updated successfully")
            else:
                logger.error("Failed to update Ragie API key")
    
    # Check and fix OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and not validate_api_key("OPENAI_API_KEY", test_openai_api_key):
        logger.warning("OpenAI API key is invalid. Please provide a new key.")
        new_key = input("Enter new OpenAI API key (or press Enter to skip): ").strip()
        if new_key:
            if update_api_key("OPENAI_API_KEY", new_key):
                # Reload environment variables
                load_dotenv()
                logger.info("OpenAI API key updated successfully")
            else:
                logger.error("Failed to update OpenAI API key")
    
    # Check and fix Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key and not validate_api_key("COHERE_API_KEY", test_cohere_api_key):
        logger.warning("Cohere API key is invalid. Please provide a new key.")
        new_key = input("Enter new Cohere API key (or press Enter to skip): ").strip()
        if new_key:
            if update_api_key("COHERE_API_KEY", new_key):
                # Reload environment variables
                load_dotenv()
                logger.info("Cohere API key updated successfully")
            else:
                logger.error("Failed to update Cohere API key")
    
    # Check and fix Serper API key
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key and not validate_api_key("SERPER_API_KEY", test_serper_api_key):
        logger.warning("Serper API key is invalid. Please provide a new key.")
        new_key = input("Enter new Serper API key (or press Enter to skip): ").strip()
        if new_key:
            if update_api_key("SERPER_API_KEY", new_key):
                # Reload environment variables
                load_dotenv()
                logger.info("Serper API key updated successfully")
            else:
                logger.error("Failed to update Serper API key")
    
    logger.info("API key fixes completed")
    return True

def fix_ragie_retrieval():
    """Fix Ragie retrieval issues."""
    logger.info("Fixing Ragie retrieval issues...")
    
    # Check if Ragie API key is valid
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("RAGIE_API_KEY is not set in the .env file")
        return False
    
    if not validate_api_key("RAGIE_API_KEY", test_ragie_api_key):
        logger.error("Ragie API key is invalid. Please fix the API key first.")
        return False
    
    # Ask if the user wants to reset the Ragie document store
    reset_ragie = input("Do you want to reset the Ragie document store? This will delete all documents. (y/n): ").strip().lower()
    if reset_ragie == 'y':
        try:
            logger.info("Resetting Ragie document store...")
            result = delete_all_documents_ragie()
            if result.get("status") == "success":
                logger.info(f"✅ Successfully deleted {result.get('deleted_count', 0)} documents from Ragie")
                logger.info(f"Failed to delete {result.get('failed_count', 0)} documents")
            else:
                logger.error(f"❌ Failed to reset Ragie document store: {result.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"❌ Error resetting Ragie document store: {e}")
            return False
    
    # Test document retrieval
    try:
        logger.info("Testing document retrieval...")
        documents = query_ragie("test query", top_k=1)
        if documents:
            logger.info(f"✅ Successfully retrieved {len(documents)} documents")
            return True
        else:
            logger.warning("⚠️ No documents retrieved. This could be normal if the document store is empty.")
            return True
    except Exception as e:
        logger.error(f"❌ Error testing document retrieval: {e}")
        return False

def fix_adaptive_router():
    """Fix adaptive router issues."""
    logger.info("Fixing adaptive router issues...")
    
    # Check if required API keys are valid
    if not validate_api_key("RAGIE_API_KEY", test_ragie_api_key):
        logger.error("Ragie API key is invalid. Please fix the API key first.")
        return False
    
    if not validate_api_key("COHERE_API_KEY", test_cohere_api_key):
        logger.error("Cohere API key is invalid. Please fix the API key first.")
        return False
    
    # Test the adaptive router
    try:
        logger.info("Testing adaptive router...")
        result = run_adaptive_rag(
            question="What is artificial intelligence?",
            clone_name="RAG_Fixer",
            clone_role="Assistant",
            persona="I am a diagnostic assistant",
            vectorstore_name="rag_fixer_test"
        )
        
        if result and "generation" in result and result["generation"] != "I'm sorry, I encountered an issue processing your request.":
            logger.info("✅ Adaptive router is working correctly")
            return True
        else:
            logger.error("❌ Adaptive router is not working correctly")
            logger.error(f"Result: {result}")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing adaptive router: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_rate_limits():
    """Provide guidance on fixing rate limit issues."""
    logger.info("Fixing rate limit issues...")
    
    logger.info("Rate limit issues can be fixed by:")
    logger.info("1. Implementing exponential backoff for retries")
    logger.info("2. Caching responses where appropriate")
    logger.info("3. Reducing the frequency of requests")
    logger.info("4. Upgrading your API plan if necessary")
    
    logger.info("\nTo implement exponential backoff, you can modify the API call functions to retry with increasing delays.")
    logger.info("Here's an example implementation for the query_ragie function:")
    
    example_code = """
import time
import random

def query_ragie_with_backoff(query, top_k=5, clone_id=None, max_retries=5):
    \"\"\"
    Query Ragie.ai with exponential backoff for rate limit handling.
    \"\"\"
    retries = 0
    while retries < max_retries:
        try:
            return query_ragie(query, top_k, clone_id)
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                wait_time = (2 ** retries) + random.random()
                logger.warning(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                # If it's not a rate limit error, re-raise it
                raise
    
    # If we've exhausted all retries
    raise Exception("Maximum retries exceeded due to rate limiting")
"""
    
    logger.info(example_code)
    
    logger.info("\nYou can implement similar backoff strategies for other API calls.")
    
    return True

def main():
    """Run the RAG fixer tool."""
    parser = argparse.ArgumentParser(description="Fix common RAG issues")
    parser.add_argument("--api-keys", action="store_true", help="Fix API key issues")
    parser.add_argument("--ragie", action="store_true", help="Fix Ragie retrieval issues")
    parser.add_argument("--router", action="store_true", help="Fix adaptive router issues")
    parser.add_argument("--rate-limits", action="store_true", help="Fix rate limit issues")
    parser.add_argument("--all", action="store_true", help="Fix all issues")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    logger.info("=" * 50)
    logger.info("Starting RAG Fixer")
    logger.info("=" * 50)
    
    results = {}
    
    # Fix API key issues
    if args.api_keys or args.all:
        results["api_keys_fixed"] = fix_api_keys()
    
    # Fix Ragie retrieval issues
    if args.ragie or args.all:
        results["ragie_retrieval_fixed"] = fix_ragie_retrieval()
    
    # Fix adaptive router issues
    if args.router or args.all:
        results["adaptive_router_fixed"] = fix_adaptive_router()
    
    # Fix rate limit issues
    if args.rate_limits or args.all:
        results["rate_limits_fixed"] = fix_rate_limits()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("RAG Fixer Summary")
    logger.info("=" * 50)
    
    for fix, result in results.items():
        status = "✅ FIXED" if result else "❌ FAILED"
        logger.info(f"{status} - {fix}")
    
    logger.info("=" * 50)
    logger.info("End of RAG Fixer")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
