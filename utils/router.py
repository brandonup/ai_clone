"""
Router module for deciding how to answer user queries
"""
import re
import logging
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Global variables to store persona-specific patterns
persona_rag_patterns = []
persona_base_llm_patterns = []

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the confidence threshold for RAG
RAG_CONFIDENCE_THRESHOLD = 0.7  # Adjust this value based on testing

def choose_route(question, rag_chunks):
    """
    Determine the appropriate route for answering a question
    using RAG confidence threshold and LLM-based routing
    
    Args:
        question: User question
        rag_chunks: List of chunks retrieved from RAG
        
    Returns:
        str: Route choice - "rag", "force_rag", "web", or "base"
    """
    # Normalize question for analysis
    q_lower = question.lower().strip()
    
    # First, check if RAG chunks exist and meet the confidence threshold
    if rag_chunks:
        rag_confidence = calculate_rag_confidence(question, rag_chunks)
        logger.info(f"RAG confidence score: {rag_confidence}")
        
        if rag_confidence >= RAG_CONFIDENCE_THRESHOLD:
            logger.info(f"Routing to 'rag' because chunks meet confidence threshold ({rag_confidence:.2f} >= {RAG_CONFIDENCE_THRESHOLD})")
            return "rag"
        else:
            logger.info(f"RAG chunks found but confidence too low ({rag_confidence:.2f} < {RAG_CONFIDENCE_THRESHOLD})")
            # Continue to other routing methods
    else:
        logger.info("No RAG chunks found")
    
    # If RAG confidence is low or no chunks found, check if it's a web search query
    if is_web_search_query(question):
        logger.info("Routing to 'web' based on LLM decision")
        return "web"
    
    # For non-web-search queries with RAG chunks (but low confidence), still use RAG
    if rag_chunks:
        logger.info("Routing to 'rag' because chunks were found (despite low confidence)")
        return "rag"
    
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
    logger.info(f"Routing to 'force_rag' as default")
    return "force_rag"

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
        # Return False if OpenAI API key is not available
        return False
    
    # Create the chat model (using a smaller/cheaper model is fine for this task)
    try:
        model = ChatOpenAI(
            temperature=0.1,  # Low temperature for consistent results
            model_name="gpt-3.5-turbo",  # Could use a smaller model to save costs
            openai_api_key=openai_api_key,
            max_tokens=50  # We only need a short response
        )
        
        # Create the prompt
        system_content = """
        You are a routing assistant for an AI Coach application. Your job is to determine if a user query requires web search.
        
        Respond ONLY with "Web Search" or "Not Web Search".
        
        Use "Web Search" for:
        - Questions about current events, news, or recent developments
        - Weather forecasts or current conditions
        - Sports scores or recent game results
        - Stock prices or market information
        - Questions containing time indicators like today, yesterday, tomorrow
        - Questions about what's currently happening or trending
        - Questions about protests, events, or incidents that happened recently
        - Questions about upcoming events, conferences, workshops, or future plans
        - Questions that mention specific dates or time periods in the future
        
        Use "Not Web Search" for:
        - General advice or coaching questions
        - Questions about concepts, theories, or timeless information
        - Personal development or self-improvement questions
        - Questions that don't require up-to-date information
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
        
        # Return True if the response contains "Web Search"
        return "web search" in result.lower()
        
    except Exception as e:
        logger.error(f"Error in LLM routing: {str(e)}")
        # Return False if LLM fails
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

def generate_persona_patterns(coach_name, coach_persona):
    """
    Generate persona-specific patterns for routing based on coach name and persona
    
    Args:
        coach_name: Name of the coach
        coach_persona: Description of the coach's persona/expertise
        
    Returns:
        dict: Dictionary containing 'rag_patterns' and 'base_llm_patterns'
    """
    global persona_rag_patterns, persona_base_llm_patterns
    
    logger.info(f"Generating persona-specific patterns for {coach_name}")
    
    # Extract key terms from the persona description
    persona_lower = coach_persona.lower()
    
    # Define common coaching domains and their related terms
    domains = {
        "fitness": ["workout", "exercise", "training", "fitness", "strength", "cardio", 
                   "weight", "muscle", "gym", "nutrition", "diet", "health"],
        
        "business": ["business", "entrepreneur", "startup", "company", "leadership", 
                    "management", "strategy", "marketing", "sales", "finance", "investment"],
        
        "life_coaching": ["goal", "motivation", "habit", "productivity", "balance", 
                         "mindset", "personal", "development", "growth", "success"],
        
        "career": ["career", "job", "interview", "resume", "workplace", "profession", 
                  "skill", "promotion", "salary", "negotiation"],
        
        "relationship": ["relationship", "communication", "conflict", "partner", 
                        "marriage", "family", "parenting", "children"],
        
        "mental_health": ["stress", "anxiety", "depression", "emotion", "therapy", 
                         "mindfulness", "meditation", "wellbeing", "wellness"]
    }
    
    # Identify which domains are relevant to this coach
    relevant_domains = []
    for domain, terms in domains.items():
        if any(term in persona_lower for term in terms):
            relevant_domains.append(domain)
            logger.info(f"Coach persona matches domain: {domain}")
    
    # If no specific domains are identified, default to general life coaching
    if not relevant_domains:
        relevant_domains = ["life_coaching"]
        logger.info("No specific domains identified, defaulting to general life coaching")
    
    # Generate RAG patterns based on relevant domains
    rag_patterns = []
    
    # Add domain-specific RAG patterns
    for domain in relevant_domains:
        if domain == "fitness":
            rag_patterns.extend([
                r'\b(workout|exercise|training) (plan|program|routine|schedule)\b',
                r'\b(diet|nutrition|meal) (plan|program|advice)\b',
                r'\bhow (to|do I) (build|gain|lose) (muscle|weight|strength)\b',
                r'\bspecific (exercise|workout|training) for (beginner|intermediate|advanced)\b'
            ])
        
        elif domain == "business":
            rag_patterns.extend([
                r'\bhow (to|do I) (start|grow|scale|manage) (a|my) (business|startup|company)\b',
                r'\b(business|marketing|sales|financial) (strategy|plan|model)\b',
                r'\b(leadership|management) (style|approach|technique|method)\b',
                r'\b(investor|investment|funding|venture capital|pitch)\b'
            ])
        
        elif domain == "life_coaching":
            rag_patterns.extend([
                r'\bhow (to|do I) (set|achieve|reach) (goals|objectives)\b',
                r'\b(morning|daily|weekly) (routine|habit|practice)\b',
                r'\b(productivity|time management|organization) (system|method|technique)\b',
                r'\b(work-life|life) balance\b'
            ])
        
        elif domain == "career":
            rag_patterns.extend([
                r'\b(resume|cv|cover letter) (advice|tips|review)\b',
                r'\b(job|interview) (preparation|tips|questions)\b',
                r'\bhow (to|do I) (negotiate|ask for) (a raise|promotion|salary)\b',
                r'\b(career|job) (change|transition|advancement)\b'
            ])
        
        elif domain == "relationship":
            rag_patterns.extend([
                r'\b(communication|conflict resolution) (skills|techniques|methods)\b',
                r'\bhow (to|do I) (improve|strengthen|fix) (my|a) relationship\b',
                r'\b(parenting|family) (advice|strategies|techniques)\b',
                r'\b(marriage|partnership) (counseling|therapy|advice)\b'
            ])
        
        elif domain == "mental_health":
            rag_patterns.extend([
                r'\b(stress|anxiety) (management|reduction|relief|techniques)\b',
                r'\b(mindfulness|meditation) (practice|technique|exercise)\b',
                r'\bhow (to|do I) (cope with|manage|overcome) (stress|anxiety|depression)\b',
                r'\b(emotional|mental) (wellbeing|health|resilience)\b'
            ])
    
    # Generate base LLM patterns (questions that don't need external knowledge)
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
        
        # Questions about the coach
        f'(who are you|tell me about yourself|what is your background|what do you do|are you {coach_name})'
    ]
    
    # Update global variables
    persona_rag_patterns = rag_patterns
    persona_base_llm_patterns = base_llm_patterns
    
    logger.info(f"Generated {len(rag_patterns)} RAG patterns and {len(base_llm_patterns)} base LLM patterns")
    
    return {
        'rag_patterns': rag_patterns,
        'base_llm_patterns': base_llm_patterns
    }
