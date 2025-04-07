"""
Entity memory module for AI Coach application
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationEntityMemory:
    """
    Maintains memory of entities mentioned in conversations
    """
    
    def __init__(self, conversation_id: Optional[str] = None, llm=None):
        """
        Initialize entity memory
        
        Args:
            conversation_id: Optional ID of the conversation to load
            llm: LLM function or object for entity extraction
        """
        self.conversation_id = conversation_id
        self.entities = {}  # Dictionary to store entity information
        self.entity_store_path = os.path.join("entity_memories", f"{conversation_id}.json") if conversation_id else None
        self.llm = llm
        self.extractor = None  # Will be initialized when needed
        
        # Create entity_memories directory if it doesn't exist
        os.makedirs("entity_memories", exist_ok=True)
        
        # Load existing entity memory if available
        if conversation_id:
            self.load_entities()
    
    def load_entities(self) -> None:
        """Load entity memory from disk"""
        if not self.entity_store_path or not os.path.exists(self.entity_store_path):
            return
            
        try:
            with open(self.entity_store_path, 'r') as f:
                self.entities = json.load(f)
                logger.info(f"Loaded {len(self.entities)} entities from {self.entity_store_path}")
        except Exception as e:
            logger.error(f"Error loading entity memory: {str(e)}")
            self.entities = {}
    
    def save_entities(self) -> None:
        """Save entity memory to disk"""
        if not self.entity_store_path:
            return
            
        try:
            with open(self.entity_store_path, 'w') as f:
                json.dump(self.entities, f, indent=2)
                logger.info(f"Saved {len(self.entities)} entities to {self.entity_store_path}")
        except Exception as e:
            logger.error(f"Error saving entity memory: {str(e)}")
    
    def get_entity_by_name(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by name
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Entity information or None if not found
        """
        # Try exact match
        if entity_name in self.entities:
            return self.entities[entity_name]
            
        # Try case-insensitive match
        entity_lower = entity_name.lower()
        for name, info in self.entities.items():
            if name.lower() == entity_lower:
                return info
                
        # Try aliases
        for name, info in self.entities.items():
            aliases = info.get("attributes", {}).get("aliases", [])
            if entity_lower in [alias.lower() for alias in aliases]:
                return info
                
        return None
    
    def update_entity(self, entity_name: str, entity_info: Dict[str, Any]) -> None:
        """
        Update or add entity information - ultra minimal version
        
        Args:
            entity_name: Name of the entity
            entity_info: Entity information
        """
        # Only store PERSON entities (user's name)
        if entity_info.get("type") != "PERSON":
            return
            
        # Skip generic entities and common words
        generic_terms = [
            "questions", "insights", "challenges", "advice", "guidance", 
            "actionable advice", "pain points", "my perspective", "your perspective",
            "ai language model", "specific goal", "entrepreneurial journey",
            "goal", "name", "my name", "ambition", "objective", "target"
        ]
        if entity_name.lower() in generic_terms:
            return
            
        # Skip short names
        if len(entity_name.strip()) <= 2:
            return
            
        # Only update if this is likely a person's name
        if entity_name in self.entities:
            # Only update attributes for existing entities
            if "attributes" in entity_info:
                # Only keep essential attributes
                essential_attributes = {}
                for key, value in entity_info.get("attributes", {}).items():
                    if key in ["interest", "goal", "role"]:
                        essential_attributes[key] = value
                
                if essential_attributes:
                    self.entities[entity_name]["attributes"].update(essential_attributes)
        else:
            # Only add new entity if it's likely the user's name
            # Create a minimal version with just the essential fields
            self.entities[entity_name] = {
                "type": "PERSON",
                "attributes": {},
                "importance": "high"
            }
            
            # Only add essential attributes
            for key, value in entity_info.get("attributes", {}).items():
                if key in ["interest", "goal", "role"]:
                    self.entities[entity_name]["attributes"][key] = value
        
        # Save changes to disk
        self.save_entities()
    
    def update_entity_from_interaction(self, user_message: str, ai_response: str) -> None:
        """
        Update entity memory based on a conversation turn - ultra minimal version
        
        Args:
            user_message: User message
            ai_response: AI response
        """
        # Lazy load the extractor
        if self.extractor is None and self.llm is not None:
            from utils.entity_extractor import HybridEntityExtractor
            self.extractor = HybridEntityExtractor(self.llm)
        
        if not self.extractor:
            logger.warning("Entity extractor not initialized, skipping entity extraction")
            return
        
        # Only extract entities from user message (not from AI response)
        # And only if the message might contain a name introduction
        if any(name_indicator in user_message.lower() for name_indicator in 
               ["my name", "i am", "i'm", "call me", "name is"]):
            user_entities = self.extractor.extract_entities(user_message)
            
            # Update entity memory (only for user entities)
            for entity_name, entity_info in user_entities.items():
                self.update_entity(entity_name, entity_info)
    
    def get_relevant_entities(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Get entities relevant to the query - ultra minimal version
        
        Args:
            query: User query
            
        Returns:
            Dictionary of relevant entities
        """
        # Only return entities if the query is asking about name or identity
        if any(term in query.lower() for term in ["my name", "who am i", "remember me", "about me"]):
            return self.entities
        
        return {}
    
    def _is_similar_mention(self, mention1: str, mention2: str) -> bool:
        """
        Check if two mentions are similar enough to be considered duplicates
        
        Args:
            mention1: First mention
            mention2: Second mention
            
        Returns:
            True if mentions are similar
        """
        # If one is a substring of the other, they're similar
        if mention1 in mention2 or mention2 in mention1:
            return True
            
        # If they share a significant number of words, they're similar
        words1 = set(mention1.lower().split())
        words2 = set(mention2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return False
            
        similarity = intersection / union
        return similarity > 0.5  # Threshold for similarity
    
    def format_entity_memories(self, entities: Dict[str, Dict[str, Any]]) -> str:
        """
        Format entity memories for inclusion in LLM context
        
        Args:
            entities: Dictionary of entities
            
        Returns:
            Formatted string of entity information
        """
        if not entities:
            return ""
        
        result = []
        for name, info in entities.items():
            entity_type = info.get("type", "UNKNOWN")
            summary = info.get("summary", "")
            attributes = info.get("attributes", {})
            
            entity_str = f"Entity: {name} (Type: {entity_type})\n"
            
            if summary:
                entity_str += f"Summary: {summary}\n"
            
            if attributes:
                entity_str += "Attributes:\n"
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, list):
                        attr_str = ", ".join(attr_value)
                        entity_str += f"- {attr_name}: {attr_str}\n"
                    else:
                        entity_str += f"- {attr_name}: {attr_value}\n"
            
            result.append(entity_str)
        
        return "\n".join(result)
