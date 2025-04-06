"""
LLM Orchestrator module for generating responses using LangChain and OpenAI
"""
import os
import re
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

def generate_answer(question, context=None, coach_name="AI Coach", persona="Persona", source="Base LLM"):
    """
    Generate an answer to a user question using LangChain and OpenAI
    with optimized context handling to avoid token limit issues
    
    Args:
        question: User question
        context: Optional context from Ragie or web search
        coach_name: Name of the coach
        persona: Coach persona description
        source: Source of the reference information (e.g., "Knowledge Base (RAG)", "Web Search (Serper.dev)")
        
    Returns:
        tuple: (Generated answer, Prompt composition)
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create the chat model
    model = ChatOpenAI(
        temperature=0.3,  # Lower temperature for more factual responses
        model_name="gpt-3.5-turbo",  # Standard chat model
        openai_api_key=openai_api_key,
        max_tokens=1024  # Limit response length to avoid token issues
    )
    
    # Use tiktoken for accurate token counting if available
    try:
        import tiktoken
        def estimate_tokens(text):
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(enc.encode(text))
        logger.info("Using tiktoken for accurate token counting")
    except ImportError:
        # Fall back to character-based estimation
        def estimate_tokens(text):
            return len(text) // 4
        logger.info("Tiktoken not available, using character-based token estimation")
    
    # Create system message with persona and instructions
    system_content = "=== COACH PERSONA ===\n"
    system_content += f"You are {coach_name} {persona}. Answer the user's question as their AI coach.\n"
    
    # Add instructions
    system_content += "\n=== INSTRUCTIONS ===\n"
    system_content += f"- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.\n"
    system_content += "- Keep answers concise unless more detail is asked, and focus on actionable advice.\n"
    system_content += "- Do not mention that you are an AI or that you're using any specific information sources.\n"
    
    # Process context to avoid token limit issues
    if context:
        # Sanitize context first to remove any problematic content
        from utils.ragie_utils import sanitize_text
        context = sanitize_text(context)
        
        # Log context size
        context_tokens = estimate_tokens(context)
        logger.info(f"Original context size: ~{context_tokens} tokens")
        
        # Calculate available token budget
        system_tokens = estimate_tokens(system_content)
        question_tokens = estimate_tokens(question)
        max_response_tokens = 1024  # From model config
        
        # OpenAI's gpt-3.5-turbo has a 16K context window
        max_total_tokens = 16000
        available_context_tokens = max_total_tokens - system_tokens - question_tokens - max_response_tokens - 100  # 100 token buffer
        
        logger.info(f"Token budget: {available_context_tokens} available for context")
        
        # If context is too large, use a smarter truncation strategy
        if context_tokens > available_context_tokens:
            logger.info(f"Context too large ({context_tokens} > {available_context_tokens}), applying smart truncation")
            
            # Try to truncate at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', context)
            truncated_context = ""
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = estimate_tokens(sentence)
                if current_tokens + sentence_tokens <= available_context_tokens:
                    truncated_context += sentence + " "
                    current_tokens += sentence_tokens
                else:
                    break
            
            if truncated_context:
                context = truncated_context.strip()
                logger.info(f"Truncated context to ~{estimate_tokens(context)} tokens using sentence boundaries")
            else:
                # If sentence-based truncation didn't work, fall back to character-based
                max_chars = available_context_tokens * 4  # Rough estimate: 4 chars per token
                context = context[:max_chars] + "..."
                logger.info(f"Truncated context to ~{estimate_tokens(context)} tokens using character-based truncation")
        
        # Add context to system message with clear separation
        system_content += f"\n=== REFERENCE INFORMATION - {source} ===\n"
        system_content += context
    
    # Create messages
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]
    
    # Log estimated token usage
    total_tokens = estimate_tokens(system_content) + estimate_tokens(question)
    logger.info(f"Estimated total input tokens: ~{total_tokens}")
    
    try:
        # Create a prompt template for LangSmith tracing
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_content),
            ("human", "{question}")
        ])
        
        # Create a chain with LangSmith tracing
        chain = (
            prompt_template 
            | model 
            | StrOutputParser()
        )
        
        # Add run metadata for better tracing
        metadata = {
            "coach_name": coach_name,
            "source": source,
            "has_context": context is not None
        }
        
        # Generate the response with tracing
        response_text = chain.invoke(
            {"question": question},
            config={"metadata": metadata, "run_name": f"AI Coach - {source}"}
        )
        
        # Create a complete prompt composition that includes both system message and user question
        # Format it with clear separators to make it easy to read
        from utils.ragie_utils import sanitize_text
        prompt_composition = """
=== SYSTEM MESSAGE ===

""" + sanitize_text(system_content) + """

=== USER QUESTION ===

""" + question
        return response_text.strip(), prompt_composition
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        
        # If we get a token limit error, try a progressive context reduction strategy
        if "maximum context length" in str(e).lower() or "token limit" in str(e).lower():
            logger.info("Token limit exceeded, implementing progressive context reduction")
            
            # Try with 50% of the context
            if context:
                # Try to identify the most relevant parts of the context
                # For simplicity, we'll use the first and last parts which often contain key information
                context_parts = context.split("\n\n")
                
                if len(context_parts) > 4:
                    # Take first 2 and last 2 paragraphs
                    reduced_context = "\n\n".join(context_parts[:2] + context_parts[-2:])
                    logger.info(f"Reduced to first 2 and last 2 paragraphs: ~{estimate_tokens(reduced_context)} tokens")
                elif len(context_parts) > 1:
                    # Take first and last paragraph
                    reduced_context = "\n\n".join([context_parts[0], context_parts[-1]])
                    logger.info(f"Reduced to first and last paragraph: ~{estimate_tokens(reduced_context)} tokens")
                else:
                    # Just take the first 50%
                    half_point = len(context) // 2
                    reduced_context = context[:half_point] + "..."
                    logger.info(f"Reduced to first 50%: ~{estimate_tokens(reduced_context)} tokens")
                
                # Create new system message with reduced context
                system_content = "=== COACH PERSONA ===\n"
                system_content += f"You are {coach_name} {persona}. Answer the user's question as their AI coach.\n"
                
                # Add instructions
                system_content += "\n=== INSTRUCTIONS ===\n"
                system_content += f"- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.\n"
                system_content += "- Keep answers concise unless more detail is asked, and focus on actionable advice.\n"
                system_content += "- Do not mention that you are an AI or that you're using any specific information sources.\n"
                
                # Add reduced context
                system_content += f"\n=== REFERENCE INFORMATION (REDUCED) - {source} ===\n"
                system_content += reduced_context
                
                messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=question)
                ]
                
                try:
                    # Create a prompt template for LangSmith tracing
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_content),
                        ("human", "{question}")
                    ])
                    
                    # Create a chain with LangSmith tracing
                    chain = (
                        prompt_template 
                        | model 
                        | StrOutputParser()
                    )
                    
                    # Add run metadata for better tracing
                    metadata = {
                        "coach_name": coach_name,
                        "source": source,
                        "has_context": True,
                        "context_type": "reduced"
                    }
                    
                    # Generate the response with tracing
                    response_text = chain.invoke(
                        {"question": question},
                        config={"metadata": metadata, "run_name": f"AI Coach - {source} (Reduced Context)"}
                    )
                    
                    from utils.ragie_utils import sanitize_text
                    prompt_composition = f"""
=== SYSTEM MESSAGE ===

{sanitize_text(system_content)}

=== USER QUESTION ===

{question}
"""
                    return response_text.strip(), prompt_composition
                except Exception as retry_error:
                    logger.error(f"Error in first retry attempt: {str(retry_error)}")
                    
                    # If still failing, try with just the most relevant paragraph
                    if "maximum context length" in str(retry_error).lower() or "token limit" in str(retry_error).lower():
                        logger.info("Still exceeding token limit, reducing to single most relevant paragraph")
                        
                        # Find paragraph with most keyword matches
                        if context_parts:
                            # Extract keywords from question
                            import re
                            keywords = set(re.findall(r'\b\w{3,}\b', question.lower()))
                            stop_words = {'the', 'and', 'for', 'with', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'this', 'that', 'have', 'has', 'had', 'not', 'are', 'about'}
                            keywords = keywords - stop_words
                            
                            # Score paragraphs by keyword matches
                            para_scores = []
                            for para in context_parts:
                                para_lower = para.lower()
                                matches = sum(1 for keyword in keywords if keyword in para_lower)
                                para_scores.append((para, matches))
                            
                            # Sort by score and take the best one
                            para_scores.sort(key=lambda x: x[1], reverse=True)
                            if para_scores and para_scores[0][1] > 0:
                                minimal_context = para_scores[0][0]
                            else:
                                # If no good match, just take the first paragraph
                                minimal_context = context_parts[0] if context_parts else ""
                        else:
                            # If no paragraphs, take a small portion
                            minimal_context = context[:500] if context else ""
                        
                        # Create new system message with minimal context
                        system_content = "=== COACH PERSONA ===\n"
                        system_content += f"You are {coach_name} {persona}. Answer the user's question as their AI coach.\n"
                        
                        # Add instructions
                        system_content += "\n=== INSTRUCTIONS ===\n"
                        system_content += f"- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.\n"
                        system_content += "- Keep answers concise unless more detail is asked, and focus on actionable advice.\n"
                        system_content += "- Do not mention that you are an AI or that you're using any specific information sources.\n"
                        
                        if minimal_context:
                            # Add minimal context
                            system_content += f"\n=== REFERENCE INFORMATION (MINIMAL) - {source} ===\n"
                            system_content += minimal_context
                        
                        messages = [
                            SystemMessage(content=system_content),
                            HumanMessage(content=question)
                        ]
                        
                        try:
                            # Create a prompt template for LangSmith tracing
                            prompt_template = ChatPromptTemplate.from_messages([
                                ("system", system_content),
                                ("human", "{question}")
                            ])
                            
                            # Create a chain with LangSmith tracing
                            chain = (
                                prompt_template 
                                | model 
                                | StrOutputParser()
                            )
                            
                            # Add run metadata for better tracing
                            metadata = {
                                "coach_name": coach_name,
                                "source": source,
                                "has_context": minimal_context != "",
                                "context_type": "minimal"
                            }
                            
                            # Generate the response with tracing
                            response_text = chain.invoke(
                                {"question": question},
                                config={"metadata": metadata, "run_name": f"AI Coach - {source} (Minimal Context)"}
                            )
                            
                            from utils.ragie_utils import sanitize_text
                            prompt_composition = """
=== SYSTEM MESSAGE ===

""" + sanitize_text(system_content) + """

=== USER QUESTION ===

""" + question
                            return response_text.strip(), prompt_composition
                        except Exception as final_error:
                            logger.error(f"Error in second retry attempt: {str(final_error)}")
                            
                            # Last resort: try without any context
                            system_content = "=== COACH PERSONA ===\n"
                            system_content += f"You are {coach_name} {persona}. Answer the user's question as their AI coach.\n"
                            
                            # Add instructions
                            system_content += "\n=== INSTRUCTIONS ===\n"
                            system_content += f"- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.\n"
                            system_content += "- Keep answers concise unless more detail is asked, and focus on actionable advice.\n"
                            system_content += "- Do not mention that you are an AI or that you're using any specific information sources.\n"
                            
                            messages = [
                                SystemMessage(content=system_content),
                                HumanMessage(content=question)
                            ]
                            
                            # Create a prompt template for LangSmith tracing
                            prompt_template = ChatPromptTemplate.from_messages([
                                ("system", system_content),
                                ("human", "{question}")
                            ])
                            
                            # Create a chain with LangSmith tracing
                            chain = (
                                prompt_template 
                                | model 
                                | StrOutputParser()
                            )
                            
                            # Add run metadata for better tracing
                            metadata = {
                                "coach_name": coach_name,
                                "source": source,
                                "has_context": False,
                                "context_type": "none"
                            }
                            
                            # Generate the response with tracing
                            response_text = chain.invoke(
                                {"question": question},
                                config={"metadata": metadata, "run_name": f"AI Coach - {source} (No Context)"}
                            )
                            
                            from utils.ragie_utils import sanitize_text
                            prompt_composition = """
=== SYSTEM MESSAGE ===

""" + sanitize_text(system_content) + """

=== USER QUESTION ===

""" + question
                            return response_text.strip() + "\n\n(Note: Answer generated without context due to token limitations)", prompt_composition
        
        # Re-raise the original error if it wasn't a token limit issue
        raise

def web_search(query):
    """
    Perform a web search using Serper.dev API with enhanced result processing
    and robust error handling
    
    Args:
        query: Search query
        
    Returns:
        str: Formatted context from search results
    """
    import logging
    from dotenv import load_dotenv
    import requests
    import json
    import re
    
    # Reload environment variables to ensure we have the latest API key
    load_dotenv(override=True)
    
    logger = logging.getLogger(__name__)
    
    # Check for API key (using the same env variable for simplicity)
    serper_api_key = os.getenv("SERPAPI_API_KEY")
    if not serper_api_key:
        logger.error("SERPAPI_API_KEY environment variable not set")
        logger.info("To use web search, add SERPAPI_API_KEY to your .env file")
        return None
    
    # Log the API key length and first/last few characters (for debugging, without revealing the full key)
    logger.info(f"Using Serper.dev API key with length: {len(serper_api_key)}")
    logger.info(f"API key starts with: {serper_api_key[:5]}... and ends with: ...{serper_api_key[-5:]}")
    
    # Optimize query for search
    search_query = query.strip()
    # Remove question marks and common question prefixes
    search_query = search_query.replace("?", "")
    prefixes_to_remove = ["what is", "who is", "how to", "where is", "when is", "why is", "can you tell me about"]
    for prefix in prefixes_to_remove:
        if search_query.lower().startswith(prefix):
            search_query = search_query[len(prefix):].strip()
    
    logger.info(f"Optimized search query: {search_query}")
    
    try:
        # Log that we're about to make the API call
        logger.info(f"Making Serper.dev request for query: {search_query}")
        
        # Serper.dev API endpoint
        url = "https://google.serper.dev/search"
        
        # Request headers
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        
        # Request payload
        payload = {
            'q': search_query,
            'gl': 'us',
            'hl': 'en',
            'num': 5
        }
        
        # Log the search parameters (excluding the API key)
        safe_payload = payload.copy()
        logger.info(f"Search parameters: {safe_payload}")
        
        # Make the request
        logger.info("Executing Serper.dev search...")
        response = requests.post(url, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info("Serper.dev search completed successfully")
            results = response.json()
        else:
            logger.error(f"Serper.dev returned an error: {response.status_code} - {response.text}")
            if response.status_code == 401:
                logger.error("Authentication failed. The Serper.dev API key may be invalid or expired.")
                logger.info("Please update your SERPAPI_API_KEY in the .env file with a valid Serper.dev key.")
            return None
        
        # Extract and format information from different result types
        formatted_results = []
        
        # 1. Extract featured snippet if available (highest quality source)
        if "answerBox" in results and "snippet" in results["answerBox"]:
            snippet = results["answerBox"]["snippet"]
            formatted_results.append(f"Featured Answer: {snippet}")
            logger.info(f"Found featured snippet: {snippet[:100]}...")
        elif "answerBox" in results and "answer" in results["answerBox"]:
            answer = results["answerBox"]["answer"]
            formatted_results.append(f"Featured Answer: {answer}")
            logger.info(f"Found featured answer: {answer[:100]}...")
        
        # 2. Extract knowledge graph information if available
        if "knowledgeGraph" in results and "description" in results["knowledgeGraph"]:
            description = results["knowledgeGraph"]["description"]
            formatted_results.append(f"Knowledge Graph: {description}")
            logger.info(f"Found knowledge graph description")
        
        # 3. Extract organic results
        if "organic" in results:
            for i, res in enumerate(results["organic"][:4]):  # Get top 4 results
                if "snippet" in res:
                    title = res.get("title", "")
                    snippet = res["snippet"]
                    formatted_results.append(f"Result {i+1} - {title}: {snippet}")
                    logger.info(f"Added organic result {i+1}")
        
        # 4. Extract news results if available (for current events)
        if "news" in results:
            for i, news in enumerate(results["news"][:2]):  # Get top 2 news results
                if "snippet" in news:
                    title = news.get("title", "")
                    date = news.get("date", "")
                    snippet = news["snippet"]
                    formatted_results.append(f"News {i+1} ({date}) - {title}: {snippet}")
                    logger.info(f"Added news result {i+1}")
        
        # Join all formatted results into a single context string
        context = "\n\n".join(formatted_results)
        
        # If we have results, return them
        if formatted_results:
            logger.info(f"Returning {len(formatted_results)} search results")
            return context
        else:
            logger.warning("No useful search results found")
            return None
            
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}")
