"""
Entity extraction module for AI Coach application
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class HybridEntityExtractor:
    """
    Hybrid approach to entity extraction using spaCy and LLM
    """
    
    def __init__(self, llm):
        """
        Initialize the entity extractor
        
        Args:
            llm: LLM function or object for entity extraction
        """
        self.llm = llm
        self.nlp = None  # Will be initialized on first use
        
        # Initialize rule-based components
        self.initialize_rules()
    
    def initialize_rules(self):
        """Initialize rule-based extraction components"""
        # Business coaching concepts
        self.business_concepts = {
            "product market fit": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["pmf", "product/market fit"],
            },
            "minimum viable product": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["mvp"],
            },
            "customer acquisition cost": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["cac", "acquisition cost"],
            },
            "lifetime value": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["ltv", "customer lifetime value", "clv"],
            },
            "business model": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["revenue model", "monetization model"],
            },
            "churn rate": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["customer churn", "attrition rate"],
            },
            "value proposition": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["unique value proposition", "uvp"],
            },
            "target market": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["target audience", "target customer"],
            },
            "go to market": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["gtm", "go-to-market strategy"],
            },
            "return on investment": {
                "type": "BUSINESS_CONCEPT",
                "aliases": ["roi", "investment return"],
            },
        }
        
        # Leadership concepts
        self.leadership_concepts = {
            "emotional intelligence": {
                "type": "LEADERSHIP_CONCEPT",
                "aliases": ["eq", "emotional quotient"],
            },
            "servant leadership": {
                "type": "LEADERSHIP_CONCEPT",
                "aliases": ["service leadership"],
            },
            "transformational leadership": {
                "type": "LEADERSHIP_CONCEPT",
                "aliases": ["transformative leadership"],
            },
            "situational leadership": {
                "type": "LEADERSHIP_CONCEPT",
                "aliases": ["adaptive leadership"],
            },
        }
        
        # Common metrics
        self.metrics_patterns = [
            (r'(\d+(?:\.\d+)?%\s+(?:increase|decrease|growth|churn|conversion|rate))', "METRIC"),
            (r'(\$\d+(?:\.\d+)?\s+(?:revenue|cost|price|value))', "FINANCIAL_METRIC"),
            (r'((?:monthly|annual|yearly|quarterly)(?:\s+recurring)?\s+revenue)', "FINANCIAL_METRIC"),
            (r'(\d+(?:\.\d+)?%)', "PERCENTAGE"),
        ]
        
        # Role patterns
        self.role_patterns = [
            (r'((?:I am|I\'m)(?: a| the)? (CEO|CTO|CFO|COO|founder|co-founder|manager|director|VP|executive|leader))', "ROLE"),
            (r'((?:my|our) (team|company|startup|business|organization))', "ENTITY"),
        ]
    
    def load_spacy(self):
        """Load spaCy model (lazy loading)"""
        if self.nlp is None:
            try:
                import spacy
                # Try to load a larger model first, fall back to smaller one if needed
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                    logger.info("Loaded spaCy model: en_core_web_lg")
                except:
                    try:
                        self.nlp = spacy.load("en_core_web_md")
                        logger.info("Loaded spaCy model: en_core_web_md")
                    except:
                        self.nlp = spacy.load("en_core_web_sm")
                        logger.info("Loaded spaCy model: en_core_web_sm")
            except ImportError:
                logger.error("spaCy not installed. Run: pip install spacy")
                logger.error("Then download a model: python -m spacy download en_core_web_lg")
                raise
    
    def extract_entities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from text using hybrid approach
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entities
        """
        entities = {}
        
        # Step 1: Apply rule-based extraction (only for critical business concepts)
        rule_entities = self.extract_with_rules(text)
        entities.update(rule_entities)
        
        # Step 2: Apply spaCy-based extraction (focused on PERSON, ORG entities)
        spacy_entities = self.extract_with_spacy(text)
        for entity_name, entity_info in spacy_entities.items():
            if entity_name not in entities:
                entities[entity_name] = entity_info
        
        # Step 3: Use LLM for remaining entity detection and enrichment
        if entities:
            # Enrich existing entities
            enriched_entities = self.enrich_with_llm(text, entities)
            entities = enriched_entities
        else:
            # Extract entities with LLM if none found so far
            llm_entities = self.extract_with_llm(text)
            entities.update(llm_entities)
        
        # Step 4: Apply importance filtering to keep only critical entities
        filtered_entities = self.filter_important_entities(text, entities)
        
        return filtered_entities
    
    def extract_with_rules(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities using rule-based approach - more selective version
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entities
        """
        entities = {}
        text_lower = text.lower()
        
        # Critical business concepts only (reduced set)
        critical_concepts = {
            "product market fit": self.business_concepts["product market fit"],
            "minimum viable product": self.business_concepts["minimum viable product"],
            "business model": self.business_concepts["business model"],
            "value proposition": self.business_concepts["value proposition"],
            "target market": self.business_concepts["target market"]
        }
        
        # Check for critical business concepts only
        for concept, details in critical_concepts.items():
            # Only extract if it's a central topic (appears multiple times or is emphasized)
            concept_count = text_lower.count(concept)
            has_alias = False
            
            # Check for aliases
            for alias in details.get("aliases", []):
                if alias in text_lower:
                    has_alias = True
                    concept_count += 1
            
            # Only store if it appears to be a central topic
            if concept_count > 1 or (concept_count == 1 and self._is_emphasized(text, concept)):
                entities[concept.title()] = {
                    "type": details["type"],
                    "mention": self._extract_relevant_sentence(text, concept),
                    "attributes": {},
                    "importance": "high"  # Mark as high importance
                }
        
        # Check for roles and entities (only direct mentions)
        for pattern, entity_type in self.role_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities[match] = {
                    "type": entity_type,
                    "mention": self._extract_relevant_sentence(text, match),
                    "attributes": {},
                    "importance": "high"  # Mark as high importance
                }
        
        return entities
    
    def extract_with_spacy(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities using spaCy - more selective version
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entities
        """
        # Lazy load spaCy
        if self.nlp is None:
            try:
                self.load_spacy()
            except:
                # If spaCy fails to load, return empty dict
                return {}
        
        entities = {}
        doc = self.nlp(text)
        
        # Extract only the most critical named entities
        critical_entity_types = ["PERSON", "ORG"]
        
        for ent in doc.ents:
            if ent.label_ in critical_entity_types:
                entity_type = {
                    "PERSON": "PERSON",
                    "ORG": "ORGANIZATION"
                }.get(ent.label_, ent.label_)
                
                # Skip common words and short entities that are likely not important
                if len(ent.text.strip()) <= 2 or ent.text.lower() in ["i", "me", "you", "we", "us", "they", "them", "it"]:
                    continue
                
                # Skip generic terms
                if ent.text.lower() in ["questions", "insights", "challenges", "advice", "guidance"]:
                    continue
                
                entities[ent.text] = {
                    "type": entity_type,
                    "mention": self._extract_relevant_sentence(text, ent.text),
                    "attributes": {},
                    "importance": "high" if ent.label_ == "PERSON" else "medium"
                }
        
        return entities
    
    def extract_with_llm(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities using LLM - more selective version
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entities
        """
        prompt = f"""
        Extract ONLY THE MOST CRITICAL entities from the following text. Be extremely selective.
        
        Text: {text}
        
        ONLY extract entities that are ABSOLUTELY ESSENTIAL to understanding the user's situation:
        - People's names (especially the user's name)
        - Specific company or organization names
        - Concrete, specific goals or objectives the user has mentioned
        - Major challenges or pain points that are central to the conversation
        
        DO NOT extract:
        - Generic concepts or terms
        - Common business jargon unless it's the central topic
        - Entities mentioned only in passing
        - Abstract concepts
        - Entities from your own responses
        - Anything that isn't a proper noun or a very specific goal
        
        Format your response as a JSON object where keys are entity names and 
        values are dictionaries with "type", "attributes", and "importance" fields.
        
        Example output:
        {{
          "John Smith": {{
            "type": "PERSON",
            "attributes": {{
              "role": "CEO",
              "company": "Acme Corp"
            }},
            "importance": "high"
          }},
          "Launch new product by Q3": {{
            "type": "GOAL",
            "attributes": {{
              "timeline": "Q3",
              "category": "product"
            }},
            "importance": "high"
          }}
        }}
        
        Be extremely selective. It's better to extract too few entities than too many.
        """
        
        try:
            response = self.llm(prompt)
            
            # Parse the JSON response
            try:
                extracted_entities = json.loads(response)
                
                # Add mention to each entity
                for entity_name, entity_info in extracted_entities.items():
                    entity_info["mention"] = text
                
                return extracted_entities
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response}")
                
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    try:
                        extracted_entities = json.loads(json_match.group(1))
                        
                        # Add mention to each entity
                        for entity_name, entity_info in extracted_entities.items():
                            entity_info["mention"] = text
                        
                        return extracted_entities
                    except:
                        pass
                
                # Return empty dict if parsing fails
                return {}
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {str(e)}")
            return {}
    
    def _is_emphasized(self, text: str, term: str) -> bool:
        """
        Check if a term appears to be emphasized in the text
        
        Args:
            text: The text to check
            term: The term to look for emphasis
            
        Returns:
            True if the term appears to be emphasized
        """
        # Check if term is at the beginning or end of a sentence
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if sentence.startswith(term.lower()) or sentence.endswith(term.lower()):
                return True
                
        # Check if term is surrounded by special characters
        emphasis_patterns = [
            rf'\*{re.escape(term)}\*',  # *term*
            rf'_{re.escape(term)}_',    # _term_
            rf'"{re.escape(term)}"',    # "term"
            rf'\b{re.escape(term)}[!?]' # term! or term?
        ]
        
        for pattern in emphasis_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def _extract_relevant_sentence(self, text: str, term: str) -> str:
        """
        Extract the most relevant sentence containing the term
        
        Args:
            text: The full text
            term: The term to find in a sentence
            
        Returns:
            The most relevant sentence or a truncated version of the text
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Find sentences containing the term
        relevant_sentences = []
        for sentence in sentences:
            if term.lower() in sentence.lower():
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Return the shortest relevant sentence to keep it concise
            return min(relevant_sentences, key=len)
        else:
            # If no sentence found, return a truncated version of the text
            max_chars = 100
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text
    
    def filter_important_entities(self, text: str, entities: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Filter entities to keep only the most important ones - ultra minimal version
        
        Args:
            text: Original text
            entities: Extracted entities
            
        Returns:
            Filtered entities
        """
        if not entities:
            return {}
            
        # Keep track of filtered entities
        filtered_entities = {}
        
        # Only keep the user's name and nothing else
        for entity_name, entity_info in entities.items():
            # Only keep PERSON entities that are likely the user's name
            if entity_info.get("type") == "PERSON":
                # Check if this might be the user's name
                if any(name_indicator in text.lower() for name_indicator in ["my name", "i am", "i'm", "call me"]):
                    # Create a minimal version of the entity
                    minimal_entity = {
                        "type": "PERSON",
                        "attributes": {},
                        "importance": "high"
                    }
                    
                    # Only keep essential attributes
                    if "attributes" in entity_info:
                        for key in ["interest", "goal", "role"]:
                            if key in entity_info["attributes"]:
                                minimal_entity["attributes"][key] = entity_info["attributes"][key]
                    
                    filtered_entities[entity_name] = minimal_entity
                    
                    # Only keep one person entity (the user)
                    break
        
        return filtered_entities
    
    def enrich_with_llm(self, text: str, entities: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Enrich entities with LLM - more selective version
        
        Args:
            text: Original text
            entities: Entities to enrich
            
        Returns:
            Enriched entities
        """
        if not entities:
            return entities
            
        entity_names = list(entities.keys())
        entity_types = {name: info.get("type", "UNKNOWN") for name, info in entities.items()}
        
        prompt = f"""
        The following entities were mentioned in this text:
        
        Text: {text}
        
        Entities:
        {json.dumps(entity_types, indent=2)}
        
        Please enrich ONLY THE MOST IMPORTANT of these entities with essential attributes and a very brief summary.
        If any of these entities seem generic or unimportant, exclude them completely.
        
        Format your response as a JSON object where keys are entity names and 
        values are dictionaries with "attributes", "summary", and "importance" fields.
        
        Example output:
        {{
          "Product Market Fit": {{
            "attributes": {{
              "definition": "When a product satisfies a strong market demand"
            }},
            "summary": "Critical milestone for startup success",
            "importance": "high"
          }}
        }}
        
        Be extremely selective. Only include entities that are truly important for remembering about this user.
        Prioritize people's names, specific goals, and concrete challenges.
        """
        
        try:
            response = self.llm(prompt)
            
            # Parse the JSON response
            try:
                enriched_data = json.loads(response)
                
                # Merge enriched data with existing entities
                for entity_name, entity_info in entities.items():
                    if entity_name in enriched_data:
                        enrichment = enriched_data[entity_name]
                        
                        # Add attributes
                        if "attributes" in enrichment:
                            entity_info.setdefault("attributes", {}).update(enrichment["attributes"])
                        
                        # Add summary
                        if "summary" in enrichment:
                            entity_info["summary"] = enrichment["summary"]
                
                return entities
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM enrichment response as JSON: {response}")
                return entities
        except Exception as e:
            logger.error(f"Error in LLM entity enrichment: {str(e)}")
            return entities
