import json
import os
import logging
import uuid

logger = logging.getLogger(__name__)

# Category to Role Mapping
CATEGORY_TO_ROLE_MAP = {
    "Characters": "Character",
    "Coaches": "Coach",
    "Therapists": "Therapist",
    "Games": "Game",
    "Historical": "Historical Figure",
    "Religion": "Religious person",
    "Animals": "Animal",
    "Discussion": "Conversation partner",
    "Comedy": "Comedian",
    "Other": "Undisclosed",
    "Authors and books": "Author",
    "Celebrities": "Celebrity",
    "Influencers": "Influencer",
    "Companions": "Companion",
    "Romantic": "Romantic interest",
    "Family": "Family member",
    "Me!": "Themselves",
    "Expert": "Expert",
    "Regular people": "Regular person",
    "CEO": "CEO",
    "Consultant": "Consultant",
    "Spokesperson": "Spokesperson"
}

CLONES_FILE = 'clones.json'

def load_clones() -> list:
    """
    Load clones from the JSON file, ensuring backward compatibility.
    - Maps 'clone_category' to 'clone_role'.
    - Removes deprecated 'enhanced_category'.
    
    Returns:
        list: A list of clone dictionaries
    """
    if not os.path.exists(CLONES_FILE):
        logger.info(f"'{CLONES_FILE}' not found. Returning empty list.")
        return []
    
    try:
        with open(CLONES_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            
            if not content:
                logger.info(f"'{CLONES_FILE}' is empty. Returning empty list.")
                return []
                
            clones_list = json.loads(content)
            
            if not isinstance(clones_list, list):
                logger.error(f"'{CLONES_FILE}' does not contain a valid JSON list. Returning empty list.")
                return []

            # --- Backward Compatibility Logic ---
            updated_clones_file = False
            for clone in clones_list:
                needs_save = False
                category = clone.get('clone_category')

                # 1. Ensure clone_role is correctly mapped from category
                if category:
                    expected_role = CATEGORY_TO_ROLE_MAP.get(category, "Undisclosed")
                    if clone.get('clone_role') != expected_role:
                        clone['clone_role'] = expected_role
                        logger.info(f"Updated 'clone_role' to mapped value '{expected_role}' for clone ID {clone.get('id')}")
                        needs_save = True
                # 2. If category missing but role exists, infer category
                elif 'clone_role' in clone and 'clone_category' not in clone:
                     # Try to find a category key that maps to the existing role value
                     inferred_category = next((cat for cat, role_val in CATEGORY_TO_ROLE_MAP.items() if role_val == clone['clone_role']), None)
                     # If we found a matching category, use it. Otherwise, use the role itself (or 'Other').
                     clone['clone_category'] = inferred_category if inferred_category else clone.get('clone_role', 'Other')
                     logger.warning(f"Set missing 'clone_category' to '{clone['clone_category']}' based on existing 'clone_role' for clone ID {clone.get('id')}")
                     # Now ensure the role is correctly mapped from the *inferred* category
                     expected_role = CATEGORY_TO_ROLE_MAP.get(clone['clone_category'], "Undisclosed")
                     if clone.get('clone_role') != expected_role:
                         clone['clone_role'] = expected_role
                         logger.info(f"Updated 'clone_role' to mapped value '{expected_role}' after inferring category for clone ID {clone.get('id')}")
                     needs_save = True
                # 3. If both are missing, set defaults
                elif 'clone_category' not in clone and 'clone_role' not in clone:
                    clone['clone_category'] = 'Other'
                    clone['clone_role'] = 'Undisclosed'
                    logger.warning(f"Set missing 'clone_category' and 'clone_role' to defaults for clone ID {clone.get('id')}")
                    needs_save = True


                # 4. Remove the old enhanced_category field if it exists
                if 'enhanced_category' in clone:
                    del clone['enhanced_category']
                    logger.info(f"Removed deprecated 'enhanced_category' field from clone ID {clone.get('id')}")
                    needs_save = True

                if needs_save:
                    updated_clones_file = True

            # Save the updated list back if any clones were modified
            if updated_clones_file:
                logger.info("Saving clones file with updated roles/removed fields.")
                # Call save_clones directly to avoid potential recursion if save_clones also calls load_clones
                try:
                    with open(CLONES_FILE, 'w', encoding='utf-8') as f_save:
                        json.dump(clones_list, f_save, indent=2)
                    logger.info(f'Successfully saved updated clones to \'{CLONES_FILE}\'.')
                except Exception as save_e:
                     logger.error(f"Error saving updated clones within load_clones: {save_e}")
                     # Decide if we should return the modified list anyway or the original list
                     # Returning modified list might be better as the data is corrected in memory
            # --- End Backward Compatibility ---

            logger.info(f'Successfully loaded {len(clones_list)} clones from \'{CLONES_FILE}\'.')
            return clones_list

    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{CLONES_FILE}'. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Error loading clones from '{CLONES_FILE}': {e}")
        return []

def save_clones(clones_list: list) -> bool:
    """
    Save clones to the JSON file.
    
    Args:
        clones_list: A list of clone dictionaries
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure data consistency before saving (optional but good practice)
        for clone in clones_list:
            if 'enhanced_category' in clone:
                 logger.warning(f"Attempting to save clone {clone.get('id')} with deprecated 'enhanced_category'. Removing before save.")
                 del clone['enhanced_category']
            category = clone.get('clone_category')
            if category:
                 expected_role = CATEGORY_TO_ROLE_MAP.get(category, "Undisclosed")
                 if clone.get('clone_role') != expected_role:
                      logger.warning(f"Correcting 'clone_role' for clone {clone.get('id')} before saving.")
                      clone['clone_role'] = expected_role

        with open(CLONES_FILE, 'w', encoding='utf-8') as f:
            json.dump(clones_list, f, indent=2)
        
        logger.info(f'Successfully saved {len(clones_list)} clones to \'{CLONES_FILE}\'.')
        return True
    except (IOError, PermissionError, TypeError) as e:
        logger.error(f"Error saving clones to '{CLONES_FILE}': {e}")
        return False
    except Exception as e:
        logger.error(f'Unexpected error saving clones: {e}')
        return False

def get_clone_by_id(clone_id: str) -> dict | None:
    """
    Get a clone by its ID.
    
    Args:
        clone_id: The ID of the clone to retrieve
        
    Returns:
        dict: The clone data if found, None otherwise
    """
    clones_list = load_clones() # load_clones now handles backward compatibility
    
    for clone in clones_list:
        if clone.get('id') == clone_id:
            logger.info(f'Found clone with ID: {clone_id}')
            return clone
    
    logger.warning(f"Clone with ID '{clone_id}' not found.")
    return None

def add_clone(new_clone_data: dict) -> bool:
    """
    Add a new clone to the clones list.
    
    Args:
        new_clone_data: Dictionary containing the clone data
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not isinstance(new_clone_data, dict) or 'id' not in new_clone_data:
        logger.error('Invalid data provided to add_clone.')
        return False
    
    # Ensure consistency before adding
    if 'enhanced_category' in new_clone_data:
        del new_clone_data['enhanced_category']
    category = new_clone_data.get('clone_category')
    if category:
        new_clone_data['clone_role'] = CATEGORY_TO_ROLE_MAP.get(category, "Undisclosed")

    clones_list = load_clones()
    
    if any(c.get('id') == new_clone_data['id'] for c in clones_list):
        logger.error(f'Attempted to add clone with duplicate ID: {new_clone_data["id"]}')
        return False
    
    clones_list.append(new_clone_data)
    logger.info(f'Adding new clone with ID: {new_clone_data["id"]}')
    return save_clones(clones_list)

def update_clone(clone_id: str, updated_data: dict) -> bool:
    """
    Update an existing clone. Ensures consistency before saving.
    
    Args:
        clone_id: The ID of the clone to update
        updated_data: Dictionary containing the updated data
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure consistency in the update data itself
    if 'enhanced_category' in updated_data:
        del updated_data['enhanced_category']
    category = updated_data.get('clone_category')
    if category:
        updated_data['clone_role'] = CATEGORY_TO_ROLE_MAP.get(category, "Undisclosed")

    clones_list = load_clones()
    updated = False
    
    for i, clone in enumerate(clones_list):
        if clone.get('id') == clone_id:
            # Update the clone data
            clones_list[i].update(updated_data)
            updated = True
            logger.info(f'Updating clone with ID: {clone_id}')
            break
    
    if not updated:
        logger.error(f'Attempted to update non-existent clone with ID: {clone_id}')
        return False
    
    # save_clones will perform a final consistency check
    return save_clones(clones_list)

def delete_clone_data(clone_id: str) -> bool:
    """
    Delete a clone by its ID.
    
    Args:
        clone_id: The ID of the clone to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    clones_list = load_clones()
    initial_length = len(clones_list)
    
    clones_list = [clone for clone in clones_list if clone.get('id') != clone_id]
    
    if len(clones_list) == initial_length:
        logger.error(f'Attempted to delete non-existent clone with ID: {clone_id}')
        return False
    
    logger.info(f'Deleting clone data with ID: {clone_id}')
    return save_clones(clones_list)
