import os
import requests # Re-added for direct Serper calls
import pprint
import json # Re-added for Serper payload
from typing import List, Optional, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.schema import Document
# from langchain_community.utilities import GoogleSearchAPIWrapper # Removed Google Search Wrapper
from langchain_cohere import ChatCohere # Use Cohere for LLM
from langchain_openai import OpenAIEmbeddings # Use OpenAI for embeddings to match vector dimensions
from langchain_core.prompts import ChatPromptTemplate
# Use Pydantic v1 for compatibility if needed, otherwise import from pydantic directly after checking versions
from pydantic.v1 import BaseModel, Field # Use pydantic v1 compatibility namespace as suggested by warning
from langchain_core.messages import HumanMessage, AIMessage # Added AIMessage
from langchain_core.output_parsers import StrOutputParser
# from langchain_qdrant import QdrantVectorStore # Removed QdrantVectorStore import
# from qdrant_client import QdrantClient # Removed QdrantClient import (global client kept for now)
from langgraph.graph import END, StateGraph, START
from .ragie_utils import query_ragie # Import the new Ragie query function

# Load environment variables
load_dotenv()

# --- Configuration ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY") # Added Ragie API Key check
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Re-added Serper Key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Removed Google API Key
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") # Removed Google CSE ID
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_COLLECTION_NAME = "clone_docs_d7ba27c0-a08b-4462-a732-2e36ffebd6e2" # Removed global default

# Optional LangSmith Tracing
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# --- Tool Definitions ---

# Serper Search Function (using requests) - Re-added
def serper_search(query: str, api_key: str) -> List[dict]:
    """Performs a web search using the Serper.dev API."""
    serper_api_endpoint = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({"q": query})
    try:
        response = requests.post(serper_api_endpoint, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
        # Extract relevant parts (adjust based on actual Serper response structure)
        organic_results = results.get("organic", [])
        formatted_results = []
        for res in organic_results[:5]: # Limit to top 5
            formatted_results.append({
                "title": res.get("title"),
                "link": res.get("link"),
                "snippet": res.get("snippet")
            })
        return formatted_results
    except requests.exceptions.RequestException as e:
        print(f"Error calling Serper API: {e}")
        return []
    except Exception as e:
        print(f"Error processing Serper response: {e}")
        return []


# --- LangGraph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        config: LangSmith config (optional)
        prompt_composition: The full prompt sent to the final LLM (optional)
        generation_attempts: Counter for tracking generation attempts (for retry logic)
        source_path: Tracks the route taken (vectorstore, web_search, llm_fallback)
        history_messages: List of BaseMessage objects for accurate prompt composition
        vectorstore_name: The specific Qdrant collection name for the current clone.
        rag_chain: Dynamically created RAG chain for the current request.
        llm_chain: Dynamically created fallback LLM chain for the current request.
        clone_name: Name of the current clone.
        clone_role: Mapped role/category of the current clone.
        persona: Persona description of the current clone.
        conversation_history_str: Formatted string of conversation history.
    """
    question: str
    generation: str
    documents: List[Document]
    config: Optional[dict]
    prompt_composition: Optional[str]
    generation_attempts: Optional[int]
    source_path: Optional[str]
    history_messages: Optional[List[Any]]
    vectorstore_name: Optional[str] # Added for per-clone collection name
    rag_chain: Optional[Any] # To hold the dynamic chain
    llm_chain: Optional[Any] # To hold the dynamic chain
    # Added clone details to state for easier access in nodes if needed
    clone_name: Optional[str]
    clone_role: Optional[str]
    persona: Optional[str]
    conversation_history_str: Optional[str]


# --- LLM and Embedding Initialization ---
llm = ChatCohere(model="command-r", temperature=0, api_key=COHERE_API_KEY)
# Use OpenAI embeddings to match the 1536-dimensional vectors
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# --- Qdrant Client Initialization (Global - Kept for now, but retrieval node will use Ragie) ---
# Initialize only the client globally, vectorstore/retriever will be per-request
try:
    print("--- Initializing Qdrant client globally (Note: Retrieval node now uses Ragie) ---")
    from qdrant_client import QdrantClient # Import locally if needed
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True, # Recommended for performance
        timeout=30.0,  # Increase timeout to 30 seconds
    )
    print("--- Successfully initialized global Qdrant client ---")
except Exception as e:
    print(f"!!! ERROR initializing global Qdrant client: {e} !!!")
    print("Qdrant client initialization failed. RAG functionality might be affected if Qdrant is used elsewhere.")
    qdrant_client = None # Set client to None if initialization fails

# --- Router Definition ---
class WebSearchTool(BaseModel): # Renamed for clarity
    """
    The internet search tool (Serper.dev). Use this for questions about current events, general knowledge,
    or topics NOT related to the primary knowledge base.
    """
    query: str = Field(description="The query to use when searching the internet with Serper.dev.")

class VectorstoreTool(BaseModel):
    """
    A vectorstore containing Paul Graham's essays. Use this for questions specifically about startups,
    entrepreneurship, business, innovation, philosophy, creativity, technology, ambition, productivity,
    growth, and fundraising based on his writings.
    """
    query: str = Field(description="The query to use when searching the Paul Graham vectorstore.")

class BaseLLMTool(BaseModel):
    """
    Use this tool for general conversation, introductions, or questions that don't fit the vectorstore or require web search.
    """
    query: str = Field(description="The user's query for the base LLM.")


router_preamble = """You are an expert at routing a user question to the most appropriate tool: a vectorstore containing knowledge documents (VectorstoreTool), a web search tool (WebSearchTool), or the base LLM (BaseLLMTool).

Follow this workflow to decide which tool to use:

1. **Vectorstore (VectorstoreTool):** Use this tool for questions that are likely to be answered by the knowledge base, including:
    - Questions about topics that would be covered in the documents stored in the system
    - Questions seeking information, explanations, or insights on any subject that doesn't explicitly require current events
    - Questions about concepts, theories, methods, history, or established facts
    - Any question that doesn't specifically need real-time or very recent information
    - ALWAYS prefer this tool first unless the question clearly requires current information or is purely conversational

2. **Web Search (WebSearchTool):** Use this tool ONLY for questions that explicitly require **current, real-time, or future information**, such as:
    - Questions explicitly mentioning "current", "latest", "recent", "today", "now", or similar time-sensitive terms
    - Questions about events that happened in the last few months
    - Questions about weather, stock prices, sports scores, or other highly time-sensitive information
    - Questions explicitly asking about upcoming events or future predictions

3. **Base LLM (BaseLLMTool):** Use this tool ONLY for:
    - Simple greetings or conversational exchanges (e.g., "Hello", "How are you?", "Thanks")
    - Questions about the AI assistant itself (e.g., "What can you do?", "Tell me about yourself")
    - Highly subjective questions that require opinions rather than factual information
    - Creative requests like writing stories or poems

When in doubt between vectorstore and web search, ALWAYS choose vectorstore first, as it contains the primary knowledge base for answering questions."""

structured_llm_router = llm.bind_tools(
    tools=[WebSearchTool, VectorstoreTool, BaseLLMTool], preamble=router_preamble # Added BaseLLMTool
)

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router


# --- Grader Definitions (Using Prompt Engineering) ---

# Retrieval Grader Prompt
retrieval_grader_prompt_template = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It is not necessary that the document contain keywords, just that the semantic meaning is relevant.

Retrieved document:
{document}

User question:
{question}

Is the document relevant to the user question? Answer ONLY 'yes' or 'no'."""
grade_prompt = ChatPromptTemplate.from_template(retrieval_grader_prompt_template)
retrieval_grader = grade_prompt | llm | StrOutputParser()

# Hallucination Grader Prompt
hallucination_grader_prompt_template = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Check if the generation refers to information that is not present in the provided facts.

Set of facts:
{documents}

LLM generation:
{generation}

Is the LLM generation grounded in the provided facts? Answer ONLY 'yes' or 'no'."""
hallucination_prompt = ChatPromptTemplate.from_template(hallucination_grader_prompt_template)
hallucination_grader = hallucination_prompt | llm | StrOutputParser()

# Answer Grader Prompt
answer_grader_prompt_template = """You are a grader assessing whether an answer addresses / resolves a question.
Check if the answer directly addresses the core question asked by the user.

User question:
{question}

LLM generation:
{generation}

Does the LLM generation address the user question? Answer ONLY 'yes' or 'no'."""
answer_prompt = ChatPromptTemplate.from_template(answer_grader_prompt_template)
answer_grader = answer_prompt | llm | StrOutputParser()


# --- Generation Chains ---

# RAG Chain with persona support
# Modified to accept clone_role
def get_rag_preamble(clone_name="AI Clone", clone_role="Assistant", persona="", conversation_history=""):
    """Get RAG preamble with persona information and conversation history."""
    preamble = f"""=== CLONE PERSONA ===
You are {clone_name}, a virtual {clone_role}. {persona}. Answer the user's question as their AI clone.

=== CONVERSATION HISTORY ===
{conversation_history}

=== INSTRUCTIONS ===
- Always respond in the tone and style of {clone_name}, with a friendly and professional tone.
- Use the following pieces of retrieved context (documents or web search results) to answer the question directly.
- Synthesize the information from the context to provide a comprehensive answer.
- If the context clearly does not contain information relevant to the question, state that you couldn't find specific information on that topic in the provided context. Do not say this if the context *is* relevant but just doesn't provide a complete answer.
- Keep answers concise unless more detail is asked, and focus on actionable advice.
- NEVER mention that you are an AI language model or that you're using any specific information sources."""
    return preamble

# LLM instance for RAG (preamble will be bound dynamically)
rag_llm = ChatCohere(model_name="command-r", temperature=0, api_key=COHERE_API_KEY)

def format_docs_for_rag(docs: List[Document]) -> List[dict]:
    """Formats LangChain Documents for Cohere RAG prompt."""
    # Added check for None in page_content
    return [{"text": doc.page_content if doc.page_content is not None else ""} for doc in docs]

def rag_prompt(x):
    formatted_docs = format_docs_for_rag(x["documents"])
    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                f"Question: {x['question']} \nAnswer: ",
                # Cohere specific format for documents
                additional_kwargs={"documents": formatted_docs},
            )
        ]
    )

# LLM Fallback Chain with persona support
# Modified to accept clone_role
def get_fallback_preamble(clone_name="AI Clone", clone_role="Assistant", persona="", conversation_history=""):
    """Get fallback preamble with persona information and conversation history."""
    preamble = f"""=== CLONE PERSONA ===
You are {clone_name}, a virtual {clone_role}. Answer the user's question as their AI clone.

=== CONVERSATION HISTORY ===
{conversation_history}

=== INSTRUCTIONS ===
- Always respond in the tone and style of {clone_name}, with a friendly and professional tone.
- Answer the question based upon your knowledge.
- Keep answers concise unless more detail is asked, and focus on actionable advice.
- NEVER mention that you are an AI language model or that you're using any specific information sources.
- Use three sentences maximum and keep the answer concise."""
    return preamble

# LLM instance for fallback (preamble will be bound dynamically)
fallback_llm = ChatCohere(model_name="command-r", temperature=0, api_key=COHERE_API_KEY)

def fallback_prompt(x):
    return ChatPromptTemplate.from_messages(
        [HumanMessage(f"Question: {x['question']} \nAnswer: ")]
    )


# --- Graph Nodes ---

def retrieve_node(state: GraphState):
    """Retrieve documents using the Ragie.ai API."""
    print("---RETRIEVE FROM RAGIE.AI---")
    question = state["question"]
    config = state.get("config") # Keep config for potential future use in query_ragie
    documents = []

    # Check if Ragie API key is available
    if not RAGIE_API_KEY:
        print("!!! ERROR: RAGIE_API_KEY not set. Cannot retrieve from Ragie.ai. !!!")
        state["documents"] = []
        state["source_path"] = "ragie_error"
        return state

    try:
        # Call the query_ragie function
        # Pass config if query_ragie is updated to use it, otherwise it's ignored
        print(f"--- Querying Ragie.ai for: '{question}' ---")
        documents = query_ragie(question) # Add config=config if needed by query_ragie
        print(f"--- Retrieved {len(documents)} documents from Ragie.ai ---")

        # Ensure documents are LangChain Documents (query_ragie should return them, but double-check)
        processed_documents = []
        if documents:
            print("--- Processing Retrieved Documents from Ragie ---")
            for i, doc in enumerate(documents):
                if isinstance(doc, Document):
                    print(f"Doc {i+1} (RAGie): {doc.page_content[:200]}...") # Print snippet
                    print(f"Doc {i+1} metadata: {doc.metadata}")
                    processed_documents.append(doc)
                else:
                    # Handle unexpected format from query_ragie if necessary
                    print(f"Doc {i+1} (not a Document object): {str(doc)}")
                    processed_documents.append(Document(page_content=str(doc), metadata={"source": "ragie.ai", "error": "unexpected_format"}))
        else:
            print(f"!!! NO DOCUMENTS RETRIEVED from Ragie.ai !!!")

        # Update state
        state["documents"] = processed_documents
        state["source_path"] = "ragie.ai" # Update source path
        return state

    except Exception as e:
        print(f"!!! ERROR retrieving from Ragie.ai: {e} !!!")
        import traceback
        traceback.print_exc()
        # Fall through to return error state

    # Return error state if retrieval failed
    state["documents"] = []
    state["source_path"] = "ragie_error" # Update source path for error
    return state


def base_llm_node(state: GraphState):
    """Generate answer using the base LLM w/o vectorstore."""
    print("---BASE LLM---") # Updated print statement
    question = state["question"]
    config = state.get("config")
    history_messages = state.get("history_messages", []) # Get history from state

    # Construct the prompt messages that will be sent
    fallback_input = {"question": question}
    # We will construct the final prompt_composition string in run_adaptive_rag
    prompt_composition_placeholder = f"Base LLM Prompt for question: {question}"
    source_path = "base_llm_cohere"

    # Get the dynamically bound LLM chain from the state if available, otherwise create it
    print("--- Checking for existing llm_chain in state ---") # New log
    current_llm_chain = state.get("llm_chain")
    if not current_llm_chain:
         print("--- llm_chain not found, creating dynamic fallback LLM chain ---") # Modified log
         try: # Add try-except around dynamic creation
             dynamic_fallback_preamble = get_fallback_preamble(
                 clone_name=state.get("clone_name", "AI Clone"),
                 clone_role=state.get("clone_role", "Assistant"),
                 persona=state.get("persona", ""),
                 conversation_history=state.get("conversation_history_str", "")
             )
             print("--- Dynamic preamble created ---") # New log
             dynamic_fallback_llm = fallback_llm.bind(preamble=dynamic_fallback_preamble)
             print("--- Dynamic LLM bound ---") # New log
             current_llm_chain = fallback_prompt | dynamic_fallback_llm | StrOutputParser()
             print("--- Dynamic LLM chain constructed ---") # New log
         except Exception as chain_creation_e:
              print(f"!!! ERROR creating dynamic fallback chain: {chain_creation_e} !!!")
              import traceback
              traceback.print_exc()
              # Cannot proceed without a chain
              state["generation"] = "Error: Failed to prepare fallback mechanism."
              state["documents"] = []
              state["source_path"] = "internal_error"
              return state
    else:
        print("--- Found existing llm_chain in state ---") # New log

    # Check if chain exists before invoking
    if not current_llm_chain:
         print("!!! ERROR: Fallback LLM chain is still missing after check/creation !!!")
         state["generation"] = "Error: Internal configuration error for fallback."
         state["documents"] = []
         state["source_path"] = "internal_error"
         return state

    # Invoke Cohere chain
    print(f"--- Invoking fallback LLM chain for question: {question} ---") # Add log
    try:
        generation = current_llm_chain.invoke(fallback_input, config=config)
        print(f"--- Fallback LLM chain invocation successful. Generation: {generation[:100]}... ---") # Add log
    except Exception as e:
        print(f"!!! ERROR during fallback LLM chain invocation: {e} !!!") # Add log
        import traceback
        traceback.print_exc()
        generation = "Error: Fallback LLM invocation failed." # Set error message

    # Update state
    state["generation"] = generation
    state["documents"] = [] # No documents used
    state["prompt_composition"] = prompt_composition_placeholder # Placeholder
    state["source_path"] = source_path # source_path defined earlier in the function
    print("--- Exiting base_llm_node ---") # Add log
    return state


def generate_node(state: GraphState):
    """Generate answer using the RAG chain."""
    print("---GENERATE RAG ANSWER---")
    question = state["question"]
    documents = state["documents"]
    config = state.get("config")
    history_messages = state.get("history_messages", []) # Get history from state

    # Initialize or increment generation attempts counter
    generation_attempts = state.get("generation_attempts", 0) + 1
    print(f"--- Generation attempt #{generation_attempts} ---")

    if not documents:
         print("---GENERATE RAG: No documents found, using base LLM---") # Updated print
         # If somehow generate is called with no docs, use base LLM
         return base_llm_node(state)

    # Ensure documents are LangChain Documents
    if isinstance(documents, list) and documents and not isinstance(documents[0], Document):
         # If web search results came in as dicts or plain text
         print(f"--- Converting non-Document objects to Documents (type: {type(documents[0])}) ---")
         documents = [Document(page_content=str(doc)) for doc in documents]
    elif isinstance(documents, Document): # Handle single document case
         print("--- Converting single Document to list ---")
         documents = [documents]
    elif not isinstance(documents, list): # Handle unexpected type
         print(f"--- Converting unexpected type to Document (type: {type(documents)}) ---")
         documents = [Document(page_content=str(documents))]

    # Print document content for debugging
    print("--- Document Content in generate_node ---")
    for i, doc in enumerate(documents):
        if isinstance(doc, Document):
            print(f"Doc {i+1} content: {doc.page_content[:100]}...")
        else:
            print(f"Doc {i+1} is not a Document object: {str(doc)[:100]}...")

    # Format documents for display in prompt composition
    formatted_docs = format_docs_for_rag(documents)
    print(f"--- Formatted {len(formatted_docs)} documents for RAG ---")

    # Construct the prompt messages that will be sent
    rag_input = {"documents": documents, "question": question}
    # We will construct the final prompt_composition string in run_adaptive_rag
    prompt_composition_placeholder = f"RAG Prompt for question: {question} with {len(documents)} documents."


    print("--- Invoking RAG chain ---")
    generation = "Error: RAG chain invocation failed." # Default error message
    try:
        # Get the dynamically bound RAG chain from the state
        current_rag_chain = state.get("rag_chain")
        if not current_rag_chain:
             # Should not happen if called after retrieve/web_search, but handle defensively
             print("!!! ERROR: RAG chain not found in state for generate_node !!!")
             raise ValueError("RAG chain missing from graph state")

        # RAG chain invocation
        generation = current_rag_chain.invoke(rag_input, config=config)
        print(f"--- Generation result: {generation[:100]}... ---")
    except Exception as e:
        print(f"!!! ERROR invoking dynamic Cohere RAG chain: {e} !!!")
        import traceback
        traceback.print_exc()
        # Keep the default error message

    # Update state
    state["documents"] = documents # Keep original LangChain documents in state
    state["generation"] = generation
    state["prompt_composition"] = prompt_composition_placeholder # Use placeholder
    state["generation_attempts"] = generation_attempts
    # source_path is preserved from input state
    return state

def grade_documents_node(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    config = state.get("config")

    print(f"--- Grading {len(documents)} documents for relevance ---")

    filtered_docs = []
    if documents: # Check if documents list is not empty
        for i, d in enumerate(documents):
            # Ensure 'd' is a Document object before accessing page_content
            if isinstance(d, Document):
                doc_content = d.page_content
                print(f"--- Document {i+1} is a Document object ---")
            else:
                # Handle cases where it might not be a Document (e.g., from web search)
                doc_content = str(d) # Convert to string as fallback
                print(f"--- Document {i+1} is NOT a Document object, converting to string ---")

            # Print a snippet of the document content for debugging
            print(f"--- Document {i+1} Content (first 100 chars): {doc_content[:100]}... ---")

            # Invoke prompt-engineered grader chain
            grade_str = retrieval_grader.invoke({"question": question, "document": doc_content}, config=config)
            print(f"--- Retrieval Grader Result for Doc {i+1}: '{grade_str}' ---")

            # Parse the string output
            if "yes" in grade_str.lower():
                print(f"---GRADE: DOCUMENT {i+1} RELEVANT---")
                # Keep the original document object if it was one
                filtered_docs.append(d if isinstance(d, Document) else Document(page_content=doc_content))
            else:
                print(f"---GRADE: DOCUMENT {i+1} NOT RELEVANT---")

    print(f"--- After grading: {len(filtered_docs)} documents found relevant out of {len(documents)} ---")
    # Update state
    state["documents"] = filtered_docs
    return state

def web_search_node(state: GraphState):
    """Web search based on the question using Serper.dev."""
    print("---WEB SEARCH (Serper)---") # Reverted print statement
    question = state["question"]
    config = state.get("config")

    if not SERPER_API_KEY: # Check for Serper key again
        print("---WEB SEARCH: SERPER_API_KEY not set. Skipping.---")
        state["documents"] = []
        state["source_path"] = "web_search_error"
        return state # Indicate error

    # Perform web search using the re-added function
    search_results = serper_search(question, SERPER_API_KEY) # Use Serper function

    # Format results into LangChain Documents
    web_docs = []
    if search_results:
        for result in search_results:
            # Adjust keys based on actual Serper response structure
            content = result.get("snippet", result.get("title", ""))
            metadata = {"source": "web_search", "url": result.get("link", "")}
            if content:
                web_docs.append(Document(page_content=content, metadata=metadata))

    if not web_docs:
         print("---WEB SEARCH: No results found.---")

    # Update state
    state["documents"] = web_docs
    state["source_path"] = "web_search"
    return state


# --- Graph Edges ---

def route_question_edge(state: GraphState):
    """Route question to web search, Ragie.ai, or LLM fallback."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    config = state.get("config")

    # Use the router to decide the primary source
    source = question_router.invoke({"question": question}, config=config)

    # If the router doesn't choose a specific tool, default to BaseLLMTool
    tool_to_use = "base_llm" # Default
    if source.tool_calls:
        datasource = source.tool_calls[0]["name"]
        if datasource == "WebSearchTool":
            tool_to_use = "web_search"
        elif datasource == "VectorstoreTool":
            # Check if Ragie is available before routing to retrieve
            if RAGIE_API_KEY:
                tool_to_use = "vectorstore" # Edge name remains 'vectorstore', points to retrieve_node
            else:
                print("---ROUTE: Router chose Vectorstore, but RAGIE_API_KEY is missing. Falling back to Base LLM.---")
                tool_to_use = "base_llm"
        elif datasource == "BaseLLMTool":
            tool_to_use = "base_llm"
        else:
            print(f"---ROUTE: Unknown tool '{datasource}', using Base LLM as fallback---")
            tool_to_use = "base_llm"
    else:
         print("---ROUTE: No specific tool called by router, using Base LLM---")
         tool_to_use = "base_llm"

    # Print final routing decision
    if tool_to_use == "web_search":
        print("---ROUTE: Question to Web Search (Serper)---")
    elif tool_to_use == "vectorstore":
        print("---ROUTE: Question to Ragie.ai (via retrieve node)---")
    elif tool_to_use == "base_llm":
        print("---ROUTE: Question to Base LLM---")

    return tool_to_use

def decide_to_generate_edge(state: GraphState):
    """Determines whether to generate an answer or fallback to web search."""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("---DECISION: No relevant documents found, using Web Search---")
        return "web_search" # Fallback to web search if vectorstore docs were irrelevant
    else:
        print("---DECISION: Relevant documents found, Generate Answer---")
        return "generate"

def grade_generation_edge(state: GraphState):
    """Determines whether the generation is grounded and answers the question."""
    print("---CHECK HALLUCINATIONS & ANSWER RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    config = state.get("config")
    generation_attempts = state.get("generation_attempts", 0)
    source_path = state.get("source_path") # Get the source path

    # Maximum number of generation attempts before falling back to web search
    MAX_GENERATION_ATTEMPTS = 3

    print(f"--- Grading generation attempt #{generation_attempts} ---")
    print(f"--- Question: {question} ---")
    print(f"--- Generation: {generation} ---")
    print(f"--- Number of documents: {len(documents)} ---")

    if not documents:
         # If generation happened without docs (e.g., web search failed, then generate called)
         # We can't check hallucinations, assume it's not supported
         print("---GRADE: No documents to check against, assuming 'not supported'---")
         # Check if the answer addresses the question anyway
         answer_grade_str = answer_grader.invoke({"question": question, "generation": generation}, config=config)
         print(f"--- Answer Grader Result (no docs): '{answer_grade_str}' ---")
         if "yes" in answer_grade_str.lower():
             print("---GRADE: Answer addresses question (no docs)---")
             return "useful"
         else:
             print("---GRADE: Answer does NOT address question (no docs)---")
             # If generation without docs isn't useful, end the graph.
             print("---DECISION: Generation without docs is not useful, ending graph.---")
             return END

    # Check Hallucinations
    # Format documents for hallucination check prompt
    doc_texts = "\n\n".join([d.page_content for d in documents if isinstance(d, Document)])
    print(f"--- Document texts for hallucination check (first 200 chars): {doc_texts[:200]}... ---")

    hallucination_grade_str = hallucination_grader.invoke({"documents": doc_texts, "generation": generation}, config=config)
    print(f"--- Hallucination Grader Result: '{hallucination_grade_str}' ---")

    if "yes" in hallucination_grade_str.lower():
        print("---GRADE: Generation IS grounded in documents---")
        # Check Answer Relevance
        answer_grade_str = answer_grader.invoke({"question": question, "generation": generation}, config=config)
        print(f"--- Answer Grader Result: '{answer_grade_str}' ---")
        if "yes" in answer_grade_str.lower():
            print("---GRADE: Generation addresses question---")
            print("---DECISION: Generation is good, returning 'useful'---")
            return "useful" # Generation is good!
        else:
            print("---GRADE: Generation does NOT address question---")
            # Check if we should retry generation or fall back to web search
            if generation_attempts < MAX_GENERATION_ATTEMPTS:
                print(f"---DECISION: Generation attempt {generation_attempts}/{MAX_GENERATION_ATTEMPTS}, retrying generation---")
                return "retry_generation"
            else:
                print(f"---DECISION: Max generation attempts ({MAX_GENERATION_ATTEMPTS}) reached.---")
                # If source was web_search, end. Otherwise, try web_search.
            if source_path == "web_search":
                # print("---DECISION: Source was web_search, ending graph.---") # Old logic
                # return END # Old logic: End here instead of looping back to web_search
                print("---DECISION: Source was web_search, falling back to base LLM---")
                return "fallback_to_base_llm" # NEW: Fallback to base LLM
            else:
                # print("---DECISION: Falling back to web search---") # Old logic
                # return "not useful" # Original fallback to web_search
                print("---DECISION: Source was vectorstore, falling back to base LLM---")
                return "fallback_to_base_llm" # NEW: Fallback to base LLM
    else:
        print("---GRADE: Generation IS NOT grounded in documents---")
        # Check if we should retry generation or fall back to web search
        if generation_attempts < MAX_GENERATION_ATTEMPTS:
            print(f"---DECISION: Generation attempt {generation_attempts}/{MAX_GENERATION_ATTEMPTS}, retrying generation---")
            return "retry_generation"
        else:
            print(f"---DECISION: Max generation attempts ({MAX_GENERATION_ATTEMPTS}) reached.---")
            # If source was web_search, end. Otherwise, try web_search.
            if source_path == "web_search":
                # print("---DECISION: Source was web_search, ending graph.---") # Old logic
                # return END # Old logic: End here instead of looping back to web_search
                print("---DECISION: Source was web_search, falling back to base LLM---")
                return "fallback_to_base_llm" # NEW: Fallback to base LLM
            else:
                # print("---DECISION: Falling back to web search---") # Old logic
                # return "not supported" # Original fallback to web_search
                print("---DECISION: Source was vectorstore, falling back to base LLM---")
                return "fallback_to_base_llm" # NEW: Fallback to base LLM


# --- Build Graph ---
workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("web_search", web_search_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("base_llm", base_llm_node) # Renamed node

# Define edges
workflow.add_conditional_edges(
    START,
    route_question_edge,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve", # Edge name stays the same, points to retrieve_node (now using Ragie)
        "base_llm": "base_llm",
    },
)

# After retrieving (from Ragie now), grade the documents
workflow.add_edge("retrieve", "grade_documents")

# After grading, decide if docs are relevant enough to generate, or fallback to web search
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate_edge,
    {
        "web_search": "web_search", # If docs were irrelevant, try web search
        "generate": "generate",   # If docs are relevant, generate answer
    },
)

# After web search, directly generate (no grading needed for web results in this flow)
# We assume web search results are generally relevant or the LLM can handle them.
workflow.add_edge("web_search", "generate")

# After generating, grade the generation
workflow.add_conditional_edges(
    "generate",
    grade_generation_edge,
    {
        "useful": END,                       # Generation is good, finish.
        "retry_generation": "generate",      # Retry generation with same documents.
        "fallback_to_base_llm": "base_llm",  # NEW: Fallback to base LLM if retries fail.
        # Keep original fallbacks in case the new state isn't returned (shouldn't happen now)
        "not useful": "base_llm",            # Fallback to base LLM instead of web_search
        "not supported": "base_llm",         # Fallback to base LLM instead of web_search
    },
)

# Base LLM node directly ends the process
workflow.add_edge("base_llm", END) # Use renamed node

# Compile the graph
app = workflow.compile()

# --- Main Function ---
# Added clone_role and vectorstore_name arguments
def run_adaptive_rag(question: str, clone_name: str = "AI Clone", clone_role: str = "Assistant", persona: str = "", vectorstore_name: Optional[str] = None, conversation_history: str = "", history_messages: Optional[List[Any]] = None, config: Optional[dict] = None) -> dict:
    """
    Runs the adaptive RAG graph for a given question, using a specific vectorstore.

    Args:
        question: The user's question.
        clone_name: Name of the clone.
        clone_role: Role/mapped category of the clone.
        persona: Clone persona description.
        vectorstore_name: The specific Qdrant collection name for this clone.
        conversation_history: The conversation history formatted as a string (for preamble).
        history_messages: The raw list of BaseMessage objects (k=5) for accurate prompt composition.
        config: Optional LangSmith configuration dictionary.

    Returns:
        A dictionary containing the final 'generation', 'documents', 'prompt_composition', and 'source_path'.
    """
    if not config:
        config = {} # Default empty config if none provided

    if not vectorstore_name:
         logger.error("!!! vectorstore_name is required for run_adaptive_rag !!!")
         # Return an error state immediately if no collection name is provided
         return {
             "generation": "Configuration error: Missing vector store identifier.",
             "documents": [],
             "prompt_composition": None,
             "source_path": "config_error"
         }

    # --- Dynamic Preamble and Chain Creation ---
    # Create the dynamic preambles using the provided arguments
    dynamic_rag_preamble = get_rag_preamble(clone_name, clone_role, persona, conversation_history)
    dynamic_fallback_preamble = get_fallback_preamble(clone_name, clone_role, persona, conversation_history)

    # Bind the global LLM instances with the dynamic preambles for this specific run
    dynamic_rag_llm = rag_llm.bind(preamble=dynamic_rag_preamble)
    dynamic_fallback_llm = fallback_llm.bind(preamble=dynamic_fallback_preamble)

    # Create the chains using the dynamically bound LLMs
    dynamic_rag_chain = rag_prompt | dynamic_rag_llm | StrOutputParser()
    dynamic_llm_chain = fallback_prompt | dynamic_fallback_llm | StrOutputParser()
    # --- End Dynamic Preamble and Chain Creation ---

    # Include history_messages, vectorstore_name, and the dynamic chains in the initial graph input state
    inputs = {
        "question": question,
        "config": config,
        "history_messages": history_messages or [],
        "vectorstore_name": vectorstore_name, # Pass the collection name
        "rag_chain": dynamic_rag_chain, # Pass the dynamic chain
        "llm_chain": dynamic_llm_chain, # Pass the dynamic chain
        # Pass clone details in case nodes need them (e.g., base_llm_node if called directly)
        "clone_name": clone_name,
        "clone_role": clone_role,
        "persona": persona,
        "conversation_history_str": conversation_history # Pass string history for base_llm_node preamble reconstruction
    }

    final_state = None
    print(f"\n--- Running Adaptive RAG for: '{question}' ---")
    try:
        # Increase recursion limit to handle complex questions
        stream_config = config.copy() if config else {}
        stream_config["recursion_limit"] = 100  # Increase from 50 to 100 to handle more complex questions

        for output in app.stream(inputs, config=stream_config):
            for key, value in output.items():
                # Node
                pprint.pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value, indent=2, width=80, depth=None)
            final_state = value # Keep track of the last state
            pprint.pprint("\n---\n")
    except Exception as e:
        print(f"!!! ERROR during graph execution/streaming: {e} !!!")
        import traceback
        traceback.print_exc()
        
        # Check if it's a recursion error
        if "Recursion limit" in str(e) or "KeyError: '__end__'" in str(e):
            print("!!! Detected recursion limit error or end node issue, falling back to base LLM !!!")
            # Fall back to base LLM directly
            try:
                # Create a fallback LLM chain
                dynamic_fallback_preamble = get_fallback_preamble(
                    clone_name=inputs.get("clone_name", "AI Clone"),
                    clone_role=inputs.get("clone_role", "Assistant"),
                    persona=inputs.get("persona", ""),
                    conversation_history=inputs.get("conversation_history_str", "")
                )
                fallback_llm_instance = fallback_llm.bind(preamble=dynamic_fallback_preamble)
                fallback_chain = fallback_prompt | fallback_llm_instance | StrOutputParser()
                
                # Generate a response using the fallback LLM
                fallback_response = fallback_chain.invoke({"question": question}, config=config)
                
                # Set a fallback state
                final_state = {
                    "question": question,
                    "generation": fallback_response,
                    "documents": [],
                    "source_path": "base_llm_fallback",
                    "prompt_composition": f"Fallback LLM Prompt: {question}"
                }
                print("!!! Successfully generated fallback response !!!")
            except Exception as fallback_e:
                print(f"!!! ERROR during fallback generation: {fallback_e} !!!")
                # If fallback fails, set a minimal error state
                final_state = {
                    "question": question,
                    "generation": f"I'm sorry, I encountered an issue processing your request. Please try again with a simpler question.",
                    "documents": [],
                    "source_path": "fallback_error",
                    "prompt_composition": None
                }
        else:
            # Set a minimal error state for other types of errors
            final_state = {
                "question": question,
                "generation": f"I'm sorry, I encountered an issue processing your request. Please try again later.",
                "documents": [],
                "source_path": "graph_error",
                "prompt_composition": None
            }

    # Extract the final generation
    # Extract the final generation and documents
    final_generation = "Sorry, I encountered an issue and couldn't generate a response."
    final_documents = []
    final_prompt_composition = None # Initialize prompt composition
    final_source_path = None # Initialize source path

    if final_state:
        final_generation = final_state.get("generation", final_generation)
        final_documents = final_state.get("documents", [])
        final_source_path = final_state.get("source_path") # Get source path from final state

        # --- Construct Final Prompt Composition String ---
        # Based on the source path, reconstruct what the final prompt likely looked like
        if final_source_path == "base_llm_cohere":
            # Reconstruct fallback prompt payload
            message_content = fallback_prompt({"question": question}).messages[0].content
            chat_history_list_for_api = []
            if history_messages:
                for msg in history_messages:
                    role = "USER" if isinstance(msg, HumanMessage) else "CHATBOT"
                    content = msg.content
                    chat_history_list_for_api.append({"role": role, "message": content})
            api_payload = {
                "message": message_content,
                "preamble": dynamic_fallback_preamble, # Use the dynamic preamble
                "chat_history": chat_history_list_for_api,
                "model": fallback_llm.model_name,
                "temperature": fallback_llm.temperature,
            }
            final_prompt_composition = f"--- Final Prompt (Base LLM) ---\n{json.dumps(api_payload, indent=2)}"

        elif final_source_path in ["ragie.ai", "web_search", "ragie_error", "vectorstore_error", "base_llm_fallback"]: # RAG path (even if docs were empty/error) - Added ragie.ai/ragie_error
             # Reconstruct RAG prompt payload
             try:
                 message_content = rag_prompt({"question": question, "documents": final_documents}).messages[0].content
             except Exception as e:
                 print(f"!!! ERROR reconstructing RAG prompt: {e} !!!")
                 message_content = f"Question: {question} \nAnswer: "
                 
             chat_history_list_for_api = []
             if history_messages:
                 for msg in history_messages:
                     try:
                         # Handle both dict and Message object formats
                         if isinstance(msg, dict) and "role" in msg and "content" in msg:
                             role = "USER" if msg["role"] == "user" else "CHATBOT"
                             content = msg["content"]
                         elif hasattr(msg, "content"):
                             # Handle Message objects
                             role = "USER" if isinstance(msg, HumanMessage) else "CHATBOT"
                             content = msg.content
                         else:
                             # Skip invalid message formats
                             print(f"!!! Skipping message with invalid format: {type(msg)} !!!")
                             continue
                         chat_history_list_for_api.append({"role": role, "message": content})
                     except Exception as msg_e:
                         print(f"!!! ERROR processing message: {msg_e} !!!")
                         # Skip this message and continue with the rest
                         continue
             cohere_docs_format = [{"text": doc.page_content if doc.page_content is not None else ""} for doc in final_documents if isinstance(doc, Document)]
             api_payload = {
                 "message": message_content,
                 "preamble": dynamic_rag_preamble, # Use the dynamic preamble
                 "chat_history": chat_history_list_for_api,
                 "documents": cohere_docs_format,
                 "model": rag_llm.model_name,
                 "temperature": rag_llm.temperature,
             }
             final_prompt_composition = f"--- Final Prompt (RAG - Source: {final_source_path}) ---\n{json.dumps(api_payload, indent=2)}"
        else:
             # Fallback if source_path is unexpected
             final_prompt_composition = f"--- Final Prompt (Unknown Source: {final_source_path}) ---\nQuestion: {question}"
        # --- End Construct Final Prompt Composition String ---

        print("--- Adaptive RAG Finished ---")
    else:
         print("--- Adaptive RAG Finished (No final state found) ---")

    # final_source_path was already assigned above
    # final_source_path = final_state.get("source_path") if final_state else None # Redundant assignment removed

    return {
        "generation": final_generation,
        "documents": final_documents,
        "prompt_composition": final_prompt_composition, # Include in return dict
        "source_path": final_source_path # Return the source path
    }


# Example Usage (for testing if run directly)
if __name__ == "__main__":
    test_question_web = "What's the latest news about AI regulations?"
    test_question_rag = "What does Paul Graham say about finding co-founders?"
    test_question_base_llm = "Hello, how are you?" # Renamed test variable

    print("\nTesting Web Search Route...")
    # Need a dummy vectorstore name for testing if Qdrant is available
    test_vectorstore_name = "test_collection_cli"
    if qdrant_client:
        # Ensure test collection exists (optional, depends on test setup)
        try:
            qdrant_client.create_collection(
                collection_name=test_vectorstore_name,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE)
            )
            print(f"Ensured test collection '{test_vectorstore_name}' exists.")
        except Exception:
             print(f"Test collection '{test_vectorstore_name}' likely already exists.")
    else:
         test_vectorstore_name = None # Cannot test RAG without client

    result_web = run_adaptive_rag(test_question_web, vectorstore_name=test_vectorstore_name)
    print(f"\nFinal Answer (Web): {result_web['generation']}")
    # print(f"Documents (Web): {result_web['documents']}")

    print("\nTesting RAG Route...")
    result_rag = run_adaptive_rag(test_question_rag, vectorstore_name=test_vectorstore_name)
    print(f"\nFinal Answer (RAG): {result_rag['generation']}")
    # print(f"Documents (RAG): {result_rag['documents']}")

    print("\nTesting Base LLM Route...") # Updated print
    result_base_llm = run_adaptive_rag(test_question_base_llm, vectorstore_name=test_vectorstore_name) # Use renamed variable
    print(f"\nFinal Answer (Base LLM): {result_base_llm['generation']}") # Updated print
