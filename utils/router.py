"""
Router module for deciding how to answer user queries
"""
import re
import logging
import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the confidence threshold for RAG
RAG_CONFIDENCE_THRESHOLD = 0.15  # Lowered from 0.2 to match the min_score in ragie_utils.py

# Global variables to store persona-specific patterns
persona_rag_patterns = []
persona_base_llm_patterns = []

def choose_route(question, ragie_chunks):
    """
    Determine the appropriate route for answering a question
    using RAG confidence threshold and LLM-based routing
    
    Args:
        question: User question
        ragie_chunks: List of chunks retrieved from Ragie
        
    Returns:
        str: Route choice - "ragie", "force_ragie", "web", or "base"
    """
    # Normalize question for analysis
    q_lower = question.lower().strip()
    
    # Special case for self-introduction questions
    self_intro_patterns = [
        r'(tell|talk).+about yourself',
        r'who are you',
        r'introduce yourself',
        r'what.+your (background|experience|expertise)',
        r'what.+your (name|role)',
        r'what do you do',
    ]
    
    for pattern in self_intro_patterns:
        if re.search(pattern, q_lower):
            logger.info(f"Routing to 'base' because self-introduction pattern matched: {pattern}")
            return "base"
    
    # Note: Persona-specific pattern matching has been removed as requested
    
    # First, check if RAG chunks exist and meet the confidence threshold
    if ragie_chunks:
        rag_confidence = calculate_rag_confidence(question, ragie_chunks)
        logger.info(f"RAG confidence score: {rag_confidence}")
        
        if rag_confidence >= RAG_CONFIDENCE_THRESHOLD:
            logger.info(f"Routing to 'ragie' because chunks meet confidence threshold ({rag_confidence:.2f} >= {RAG_CONFIDENCE_THRESHOLD})")
            return "ragie"
        else:
            logger.info(f"RAG chunks found but confidence too low ({rag_confidence:.2f} < {RAG_CONFIDENCE_THRESHOLD})")
            # Continue to other routing methods
    else:
        logger.info("No RAG chunks found")
    
    # If RAG confidence is low or no chunks found, check if it's a web search query
    web_search_needed = is_web_search_query(question)
    if web_search_needed:
        logger.info("Routing to 'web' based on LLM decision")
        return "web"
    else:
        logger.info("LLM decided this query does NOT need web search")
    
    # For non-web-search queries with RAG chunks (but low confidence), still use RAG
    if ragie_chunks:
        logger.info("Routing to 'ragie' because chunks were found (despite low confidence)")
        return "ragie"
    
    # Check for base LLM indicators (questions that don't need external knowledge)
    base_llm_patterns = [
        # General advice or opinion questions
        r'^(what (do|would) you|how (do|would|can) you)',
        r'^(can you|could you) (help|advise|suggest)',
        
        # Hypothetical scenarios
        r'(if|imagine|suppose|what if)',
        
        # Personal coaching questions
        r'(how (can|should) i|what (can|should) i)',
        r'(advice|suggestion|tip|help) (for|with|on)',
        
        # Simple factual questions that a general LLM should know
        r'^(what is|who is|how does) (a|an|the)',
    ]
    
    # Check if any base LLM patterns match
    for pattern in base_llm_patterns:
        if re.search(pattern, q_lower):
            logger.info(f"Routing to 'base' because pattern matched: {pattern}")
            return "base"
    
    # Default to trying RAG again with a modified query
    # This implements the advanced RAG optimization approach
    logger.info(f"Routing to 'force_ragie' as default")
    return "force_ragie"

def calculate_rag_confidence(question, chunks):
    """
    Calculate a confidence score for RAG chunks
    
    Args:
        question: User question
        chunks: List of text chunks from RAG
        
    Returns:
        float: Confidence score between 0 and 1
    """
    # Use the existing relevance scoring function
    # Extract key terms from the question
    key_terms = extract_key_terms(question)
    logger.info(f"Key terms extracted: {key_terms}")
    
    if not key_terms or not chunks:
        return 0.0
    
    # Calculate term frequency across all chunks
    term_matches = {}
    for term in key_terms:
        term_matches[term] = 0
        
    # Count how many chunks contain each key term
    for chunk in chunks:
        chunk_lower = chunk.lower()
        for term in key_terms:
            if term in chunk_lower:
                term_matches[term] += 1
    
    # Calculate the average term frequency across chunks
    total_matches = sum(term_matches.values())
    max_possible_matches = len(key_terms) * len(chunks)
    
    # Calculate term coverage (what percentage of terms appear at least once)
    terms_covered = sum(1 for term, count in term_matches.items() if count > 0)
    term_coverage = terms_covered / len(key_terms) if key_terms else 0
    
    # Calculate chunk relevance (what percentage of chunks contain at least some key terms)
    relevant_chunks = 0
    for chunk in chunks:
        if contains_key_terms(chunk, key_terms):
            relevant_chunks += 1
    chunk_relevance = relevant_chunks / len(chunks) if chunks else 0
    
    # Combine metrics into a single confidence score
    # Weight the metrics based on importance
    term_frequency_weight = 0.3
    term_coverage_weight = 0.4
    chunk_relevance_weight = 0.3
    
    term_frequency_score = total_matches / max_possible_matches if max_possible_matches > 0 else 0
    
    confidence_score = (
        term_frequency_score * term_frequency_weight +
        term_coverage * term_coverage_weight +
        chunk_relevance * chunk_relevance_weight
    )
    
    logger.info(f"RAG confidence calculation:")
    logger.info(f"  - Term frequency: {term_frequency_score:.2f} (weight: {term_frequency_weight})")
    logger.info(f"  - Term coverage: {term_coverage:.2f} (weight: {term_coverage_weight})")
    logger.info(f"  - Chunk relevance: {chunk_relevance:.2f} (weight: {chunk_relevance_weight})")
    logger.info(f"  = Final confidence: {confidence_score:.2f}")
    
    return confidence_score

def is_web_search_query(question):
    """
    Use the LLM to determine if a query requires web search
    
    Args:
        question: User question
        
    Returns:
        bool: True if the query should use web search
    """
    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        # Fall back to pattern matching
        return check_time_sensitive_patterns(question.lower())
    
    # Create the chat model (using a smaller/cheaper model is fine for this task)
    try:
        model = ChatOpenAI(
            temperature=0.1,  # Low temperature for consistent results
            model_name="gpt-3.5-turbo",  # Standard chat model
            openai_api_key=openai_api_key,
            max_tokens=50  # We only need a short response
        )
        
        # Create the prompt
        system_content = """
        You are a routing assistant for an AI Coach application. Your job is to determine if a user query requires up-to-date or real-time information from the web.
        
        Respond ONLY with "Web Search" or "Not Web Search".
        
        Consider the following:
        1. **Time Sensitivity**: Does the user need the most recent information (e.g., current events, news, sports scores, weather, stock prices, product availability)?
        2. **Knowledge Cutoff**: Could the query refer to information that may not be in your existing knowledge (e.g., anything that happened after your cutoff date)?
        3. **Explicit Time References**: Does the query mention “today,” “yesterday,” “this week,” “recently,” or other indicators that the user wants current data?
        4. **Location-Based or Dynamic Data**: Isf the user asking for data that changes often (local events, traffic updates, store hours, product inventory, etc.)?

        Use "Web Search" if the query:
        - Asks about current or recent events, breaking news, or up-to-date statistics
        - Mentions or implies the need for real-time data (weather forecasts, stock/crypto prices, sports scores)
        - Includes explicit time references like “today,” “yesterday,” “tomorrow,” “in the last month,” etc.
        - Requests location-specific or dynamic information (e.g., store hours, local traffic, conference dates)
        - Asks about trending topics, social media updates, or anything that could change rapidly
        - Involves topics that are likely to have changed since your knowledge cutoff (e.g., new product releases, policy updates, trending topics)

        Use "Not Web Search" if the query:
        - Asks for general advice, coaching, or self-improvement
        - Seeks explanations of concepts, theories, or other timeless information
        - Is hypothetical or does not depend on the latest data
        - Refers to well-established knowledge (historical events, widely recognized facts) that likely hasn't changed

        Respond ONLY with:
        - **"Web Search"** if the user query fits any time-sensitive or real-time criteria.
        - **"Not Web Search"** otherwise.
        """
        
        
        # Create messages
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Query: {question}")
        ]
        
        # Get the response
        logger.info(f"Sending query to LLM for routing decision: {question}")
        response = model(messages)
        result = response.content.strip()
        
        logger.info(f"LLM routing decision for '{question}': {result}")
        
        # Return True if the response contains "Web Search", False if it contains "Not Web Search"
        if "not web search" in result.lower():
            logger.info("LLM decided this query does NOT need web search")
            return False
        elif "web search" in result.lower():
            logger.info("LLM decided this query needs web search")
            return True
        else:
            # If the response doesn't match expected format, fall back to pattern matching
            logger.warning(f"Unexpected LLM response format: {result}. Falling back to pattern matching.")
            return check_time_sensitive_patterns(question.lower())
        
    except Exception as e:
        logger.error(f"Error in LLM routing: {str(e)}")
        # Fall back to pattern matching if LLM fails
        logger.info("Falling back to pattern matching for routing decision")
        return check_time_sensitive_patterns(question.lower())

def check_time_sensitive_patterns(q_lower):
    """
    Check if a query is time-sensitive or requires current information
    
    Args:
        q_lower: Lowercase question
        
    Returns:
        bool: True if it's time-sensitive
    """
    # Enhanced patterns for time-sensitive queries
    time_sensitive_patterns = [
        # Time indicators (high priority)
        r'\b(today|tomorrow|yesterday|tonight|this morning|this afternoon|this evening)\b',
        r'\b(current|latest|recent|now|right now|at the moment)\b',
        r'\b(this week|this month|this year|next week|next month|next year)\b',
        
        # News and events
        r'\b(news|headline|breaking|update|development|announcement)\b',
        r'\b(happened|occurring|taking place|going on|event)\b',
        
        # Sports and entertainment
        r'\b(score|game|match|playing|showing|performance|concert)\b',
        
        # Market information
        r'\b(stock price|market|rate|trading at|exchange rate)\b',
        
        # Specific dates that suggest current information
        r'\b(in 2024|in 2025|2024|2025)\b',
        
        # Questions about protests, events, or incidents
        r'\b(protest|event|incident|happening|occurred)\b',
        
        # Weather terms
        r'\b(weather|forecast|temperature|rain|snow|sunny|cloudy)\b',
    ]
    
    for pattern in time_sensitive_patterns:
        if re.search(pattern, q_lower):
            logger.info(f"Time-sensitive pattern matched: {pattern}")
            return True
            
    return False

def extract_key_terms(question):
    """
    Extract important terms from the question
    
    Args:
        question: User question
        
    Returns:
        list: List of key terms
    """
    q_lower = question.lower()
    
    # Remove common stop words
    stop_words = ["what", "where", "when", "how", "why", "is", "are", "the", "a", "an", "in", 
                 "on", "at", "to", "for", "with", "about", "like", "going", "be", "will", 
                 "can", "could", "would", "should", "do", "does", "did", "has", "have", "had"]
    
    for word in stop_words:
        q_lower = q_lower.replace(f" {word} ", " ")
    
    # Split into words and filter out short words
    terms = [word for word in q_lower.split() if len(word) > 3]
    
    # Remove duplicates while preserving order
    unique_terms = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)
    
    return unique_terms

def contains_key_terms(chunk, key_terms):
    """
    Check if a chunk contains enough key terms to be considered relevant
    
    Args:
        chunk: Text chunk from RAG
        key_terms: List of key terms from the question
        
    Returns:
        bool: True if chunk contains enough key terms
    """
    chunk_lower = chunk.lower()
    matches = 0
    
    for term in key_terms:
        if term in chunk_lower:
            matches += 1
    
    # Require at least 2 key terms or 30% of the terms, whichever is less
    min_required = min(2, max(1, int(len(key_terms) * 0.3)))
    
    return matches >= min_required

def generate_persona_patterns(coach_name, persona_description):
    """
    Generate routing patterns based on the coach's persona description
    
    Args:
        coach_name: Name of the coach
        persona_description: Description of the coach's persona
        
    Returns:
        dict: Dictionary containing empty patterns for RAG and base LLM routing
    """
    # Pattern matching has been removed as requested
    # Always return empty patterns
    logger.info(f"Pattern matching disabled. Returning empty patterns for coach: {coach_name}")
    return {
        "rag_patterns": [],
        "base_llm_patterns": []
    }
