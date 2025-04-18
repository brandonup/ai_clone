#!/usr/bin/env python3
"""
Firestore Data Puller

This script pulls all clone data from Firestore and saves it locally for use in RAG diagnostics.
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("firestore_pull.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("firestore_pull")

# Import project modules
sys.path.append('backend')
try:
    from utils.clone_manager import load_clones, get_clone_by_id
except ImportError as e:
    logger.error(f"Error importing project modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def pull_all_clones(output_file="clones_data.json"):
    """Pull all clone data from Firestore and save it to a local file."""
    logger.info("Pulling all clone data from Firestore...")
    
    try:
        # Load all clones from Firestore
        clones = load_clones()
        
        if not clones:
            logger.warning("No clones found in Firestore.")
            return False
        
        logger.info(f"Successfully pulled {len(clones)} clones from Firestore.")
        
        # Save the clones to a local file
        with open(output_file, 'w') as f:
            json.dump(clones, f, indent=2)
        
        logger.info(f"Saved clone data to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error pulling clone data from Firestore: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def pull_specific_clone(clone_id, output_file=None):
    """Pull a specific clone from Firestore and save it to a local file."""
    logger.info(f"Pulling clone {clone_id} from Firestore...")
    
    try:
        # Get the clone from Firestore
        clone = get_clone_by_id(clone_id)
        
        if not clone:
            logger.warning(f"Clone {clone_id} not found in Firestore.")
            return False
        
        logger.info(f"Successfully pulled clone {clone_id} from Firestore.")
        
        # Save the clone to a local file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(clone, f, indent=2)
            
            logger.info(f"Saved clone data to {output_file}")
        
        return clone
    except Exception as e:
        logger.error(f"Error pulling clone {clone_id} from Firestore: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def update_diagnose_rag_script(clones_file="clones_data.json"):
    """Update the diagnose_rag.py script to use the local clone data."""
    logger.info("Updating diagnose_rag.py to use local clone data...")
    
    try:
        # Check if the clones file exists
        if not os.path.exists(clones_file):
            logger.error(f"Clones file {clones_file} not found.")
            return False
        
        # Check if the diagnose_rag.py file exists
        if not os.path.exists("diagnose_rag.py"):
            logger.error("diagnose_rag.py file not found.")
            return False
        
        # Read the current diagnose_rag.py file
        with open("diagnose_rag.py", 'r') as f:
            content = f.read()
        
        # Check if the file already has the clone data loading code
        if "load_clone_data" in content:
            logger.info("diagnose_rag.py already has clone data loading code.")
            return True
        
        # Find the right place to insert the code
        import_section_end = content.find("# Configure logging")
        if import_section_end == -1:
            logger.error("Could not find the import section in diagnose_rag.py.")
            return False
        
        # Create the code to insert
        clone_data_code = """
# Load clone data from local file
def load_clone_data(file_path="clones_data.json"):
    \"\"\"Load clone data from a local file.\"\"\"
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading clone data from {file_path}: {e}")
        return []

# Get a random clone ID for testing
def get_random_clone_id():
    \"\"\"Get a random clone ID from the loaded clone data.\"\"\"
    clones = load_clone_data()
    if not clones:
        return None
    import random
    return random.choice(clones)["id"]
"""
        
        # Insert the code after the import section
        new_content = content[:import_section_end] + clone_data_code + content[import_section_end:]
        
        # Update the test_adaptive_router function to use a real clone ID
        test_router_start = new_content.find("def test_adaptive_router(")
        if test_router_start == -1:
            logger.error("Could not find the test_adaptive_router function in diagnose_rag.py.")
            return False
        
        # Find the function body
        function_body_start = new_content.find(":", test_router_start) + 1
        
        # Find where the test cases are defined
        test_cases_start = new_content.find("test_cases =", function_body_start)
        if test_cases_start == -1:
            logger.error("Could not find the test_cases definition in test_adaptive_router function.")
            return False
        
        # Insert code to get a random clone ID before the test cases
        clone_id_code = """
    # Get a random clone ID for testing
    clone_id = get_random_clone_id()
    if not clone_id:
        logger.warning("No clone IDs available for testing. Using 'rag_diagnostics_test' as fallback.")
        clone_id = "rag_diagnostics_test"
    else:
        logger.info(f"Using clone ID {clone_id} for testing")
"""
        
        # Insert the code before the test cases
        new_content = new_content[:test_cases_start] + clone_id_code + new_content[test_cases_start:]
        
        # Update the run_adaptive_rag call to use the real clone ID
        for test_case in ["vectorstore", "web_search", "base_llm"]:
            run_rag_call = new_content.find(f'vectorstore_name="rag_diagnostics_test"')
            if run_rag_call != -1:
                new_content = new_content.replace(
                    f'vectorstore_name="rag_diagnostics_test"',
                    f'vectorstore_name=clone_id'
                )
        
        # Write the updated content back to the file
        with open("diagnose_rag.py", 'w') as f:
            f.write(new_content)
        
        logger.info("Successfully updated diagnose_rag.py to use local clone data.")
        return True
    except Exception as e:
        logger.error(f"Error updating diagnose_rag.py: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run the Firestore data puller."""
    parser = argparse.ArgumentParser(description="Pull clone data from Firestore")
    parser.add_argument("--clone-id", help="Pull a specific clone by ID")
    parser.add_argument("--output", default="clones_data.json", help="Output file path")
    parser.add_argument("--update-script", action="store_true", help="Update diagnose_rag.py to use local clone data")
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Starting Firestore Data Puller")
    logger.info("=" * 50)
    
    success = False
    
    if args.clone_id:
        # Pull a specific clone
        clone = pull_specific_clone(args.clone_id, args.output)
        success = bool(clone)
    else:
        # Pull all clones
        success = pull_all_clones(args.output)
    
    if success and args.update_script:
        # Update the diagnose_rag.py script
        update_diagnose_rag_script(args.output)
    
    logger.info("=" * 50)
    logger.info("Firestore Data Puller Complete")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
