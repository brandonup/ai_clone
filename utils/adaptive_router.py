import os
import requests
import pprint
import json # Added import
from typing import List, Optional, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_cohere import ChatCohere # Use Cohere for LLM
from langchain_openai import OpenAIEmbeddings # Use OpenAI for embeddings to match vector dimensions
from langchain_core.prompts import ChatPromptTemplate
# Use Pydantic v1 for compatibility if needed, otherwise import from pydantic directly after checking versions
from pydantic.v1 import BaseModel, Field # Use pydantic v1 compatibility namespace as suggested by warning
from langchain_core.messages import HumanMessage, AIMessage # Added AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore # Import newer non-deprecated class
from qdrant_client import QdrantClient
from langgraph.graph import END, StateGraph, START

# Load environment variables
load_dotenv()

# --- Configuration ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Corrected variable name
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "ai_coach_docs" # As specified by user

# Optional LangSmith Tracing
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# --- Tool Definitions ---

# Serper Search Function (using requests)
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
    """
    question: str
    generation: str
    documents: List[Document]
    config: Optional[dict]
    prompt_composition: Optional[str]
    generation_attempts: Optional[int]
    source_path: Optional[str] # Added to track the source path
    history_messages: Optional[List[Any]] # Added to pass raw messages


# --- LLM and VectorStore Initialization ---
llm = ChatCohere(model="command-r", temperature=0, api_key=COHERE_API_KEY)
# Use OpenAI embeddings to match the 1536-dimensional vectors in the Qdrant collection
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Initialize Qdrant Client and Vectorstore globally
try:
    print("--- Initializing Qdrant client globally ---")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True, # Recommended for performance
        timeout=30.0,  # Increase timeout to 30 seconds
    )
    print("--- Initializing Qdrant vectorstore globally ---")
    vectorstore = QdrantVectorStore(
        client=qdrant_client, # Pass the initialized client
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings, # Use 'embedding' (singular)
        content_payload_key="text", # Explicitly tell Qdrant where the content is
        metadata_payload_key="metadata" # Assuming metadata is stored under 'metadata' key
    )
    # Initialize retriever without extra search_kwargs
    retriever = vectorstore.as_retriever()
    print("--- Successfully initialized global Qdrant vectorstore and retriever ---")
except Exception as e:
    print(f"!!! ERROR initializing global Qdrant components: {e} !!!")
    print("Continuing without Qdrant vectorstore. RAG functionality will be limited.")
    vectorstore = None
    retriever = None
    # Don't re-raise the exception to allow the app to start without Qdrant

# --- Router Definition ---
class WebSearchTool(BaseModel): # Renamed for clarity
    """
    The internet search tool (Serper.dev). Use this for questions about current events, general knowledge,
    or topics NOT related to Paul Graham's essays on startups, entrepreneurship, business, innovation, etc.
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


router_preamble = """You are an expert at routing a user question to the most appropriate tool: a vectorstore containing Paul Graham's essays (VectorstoreTool), a web search tool (WebSearchTool), or the base LLM (BaseLLMTool).

Follow this workflow to decide which tool to use:

1. **Vectorstore (VectorstoreTool):** Use this tool *only* for questions specifically asking about **Paul Graham's views, writings, or insights** on the following topics:
    - startups, entrepreneurship, business, innovation, technology
    - fundraising, growth, productivity, ambition
    - philosophy (his specific views), creativity
    - personal development, self-improvement (as discussed in his essays)
    - general advice or coaching questions *based on his known work*.
    - Use this also for questions about these topics that do *not* require up-to-the-minute information.
2. **Web Search (WebSearchTool):** Use this tool for questions requiring **current, real-time, or future information**, such as:
    - Current events, news, or recent developments (e.g., recent tech news, market changes like the SVB crisis)
    - Information dependent on time (e.g., "today's weather", "next week's conference", "yesterday's stock price", "IPO tomorrow")
    - Specific factual lookups not covered by the vectorstore (e.g., "best meeting venue in Palo Alto", "current sports scores").
    - Questions explicitly asking about recent events or future plans.
3. **Base LLM (BaseLLMTool):** Use this tool for:
    - **General conversation** (e.g., "Hello", "How are you?", "Thanks").
    - Introduction questions (e.g., "Tell me about yourself", "Whats your background?").
    - Questions that fall outside the specific scope of the vectorstore (Paul Graham's work) and do not require current information from the web search.
    - Abstract questions not directly related to the vectorstore topics.

Choose the single most appropriate tool based on the user's question."""

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
def get_rag_preamble(coach_name="AI Coach", persona="", conversation_history=""):
    """Get RAG preamble with persona information and conversation history."""
    preamble = f"""=== COACH PERSONA ===
You are {coach_name} {persona}. Answer the user's question as their AI coach.

=== CONVERSATION HISTORY ===
{conversation_history}

=== INSTRUCTIONS ===
- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.
- Use the following pieces of retrieved context (documents or web search results) to answer the question directly.
- Synthesize the information from the context to provide a comprehensive answer.
- If the context clearly does not contain information relevant to the question, state that you couldn't find specific information on that topic in the provided context. Do not say this if the context *is* relevant but just doesn't provide a complete answer.
- Keep answers concise unless more detail is asked, and focus on actionable advice.
- NEVER mention that you are an AI language model or that you're using any specific information sources."""
    return preamble

# Initialize with default preamble (will be updated when run_adaptive_rag is called)
rag_llm = ChatCohere(model_name="command-r", temperature=0, api_key=COHERE_API_KEY).bind(
    preamble=get_rag_preamble()
)

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

rag_chain = rag_prompt | rag_llm | StrOutputParser()

# LLM Fallback Chain with persona support
def get_fallback_preamble(coach_name="AI Coach", persona="", conversation_history=""):
    """Get fallback preamble with persona information and conversation history."""
    preamble = f"""=== COACH PERSONA ===
You are {coach_name} {persona}. Answer the user's question as their AI coach.

=== CONVERSATION HISTORY ===
{conversation_history}

=== INSTRUCTIONS ===
- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.
- Answer the question based upon your knowledge.
- Keep answers concise unless more detail is asked, and focus on actionable advice.
- NEVER mention that you are an AI language model or that you're using any specific information sources.
- Use three sentences maximum and keep the answer concise."""
    return preamble

# Initialize with default preamble (will be updated when run_adaptive_rag is called)
fallback_llm = ChatCohere(model_name="command-r", temperature=0, api_key=COHERE_API_KEY).bind(
    preamble=get_fallback_preamble()
)

def fallback_prompt(x):
    return ChatPromptTemplate.from_messages(
        [HumanMessage(f"Question: {x['question']} \nAnswer: ")]
    )

llm_chain = fallback_prompt | fallback_llm | StrOutputParser()


# --- Graph Nodes ---

def retrieve_node(state: GraphState):
    """Initialize retriever and retrieve documents from Qdrant vectorstore."""
    print("---RETRIEVE FROM VECTORSTORE (Node Initialization)---")
    question = state["question"]
    config = state.get("config")
    documents = []

    # If global retriever is available, use it
    if retriever is not None:
        try:
            # Use the global retriever
            print("--- Using global retriever ---")
            documents = retriever.invoke(question, config=config)
            print(f"--- Retrieved {len(documents)} documents ---")
            
            # Debug: Print document contents and ensure page_content is populated
            processed_documents = []
            if documents:
                print("--- Processing Retrieved Documents ---")
                for i, doc in enumerate(documents):
                    if isinstance(doc, Document):
                        # Try to get content from page_content first
                        content = doc.page_content
                        # If page_content is empty, try getting it from metadata payload['text']
                        if not content and doc.metadata and 'payload' in doc.metadata and isinstance(doc.metadata['payload'], dict) and 'text' in doc.metadata['payload']:
                            print(f"--- Doc {i+1}: page_content empty, using payload['text'] ---")
                            content = doc.metadata['payload']['text']
                            doc.page_content = content # Update the document object
                        elif not content:
                             print(f"--- Doc {i+1}: page_content and payload['text'] are empty or missing ---")
                        
                        print(f"Doc {i+1} (Document object): {content[:200]}...") # Print snippet
                        print(f"Doc {i+1} metadata: {doc.metadata}")
                        processed_documents.append(doc) # Add the potentially updated doc
                    else:
                        # Handle non-Document objects if necessary (though retriever should return Documents)
                        print(f"Doc {i+1} (not a Document object): {str(doc)}")
                        processed_documents.append(Document(page_content=str(doc))) # Convert to Document

                # Print document types
                print("--- Document Types After Processing ---")
                for i, doc in enumerate(processed_documents):
                    print(f"Doc {i+1} type: {type(doc)}")
            else:
                print("!!! NO DOCUMENTS RETRIEVED !!!")
                
            return {"documents": processed_documents, "question": question, "source_path": "vectorstore"} # Set source_path
        except Exception as e:
            print(f"!!! ERROR using global retriever: {e} !!!")
            import traceback
            traceback.print_exc()
            # Fall through to try local initialization
    
    # If global retriever is not available or failed, try to initialize locally
    try:
        # Initialize components within the node execution context
        print("--- Initializing Qdrant client locally for retrieval ---")
        local_qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=30.0,  # Increase timeout to 30 seconds
        )

        # Use the globally defined OpenAI embeddings
        print(f"--- Using embedding model: {embeddings.model} ---")

        print("--- Initializing Qdrant vectorstore locally for retrieval ---")
        local_vectorstore = QdrantVectorStore.from_existing_collection(
            client=local_qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )

        local_retriever = local_vectorstore.as_retriever()
        print("--- Successfully initialized components locally for retrieval ---")

        # Perform retrieval
        documents = local_retriever.invoke(question, config=config)
        print(f"--- Retrieved {len(documents)} documents ---")

    except Exception as e:
        print(f"!!! ERROR during retrieval node execution: {e} !!!")
        print("!!! Falling back to web search or LLM !!!")
        import traceback
        traceback.print_exc()
        documents = [] # Return empty list on error

    return {"documents": documents, "question": question, "source_path": "vectorstore_error"} # Indicate error

def base_llm_node(state: GraphState):
    """Generate answer using the base LLM w/o vectorstore."""
    print("---BASE LLM---") # Updated print statement
    question = state["question"]
    config = state.get("config")
    history_messages = state.get("history_messages", []) # Get history from state

    # Construct the prompt messages that will be sent
    fallback_input = {"question": question}
    prompt_value = fallback_prompt(fallback_input)
    # Construct the likely API payload for documentation
    global fallback_preamble, fallback_llm, llm_chain # Use existing Cohere chain
    prompt_value = fallback_prompt(fallback_input)
    # Extract message content
    message_content = ""
    if prompt_value.messages and isinstance(prompt_value.messages[0], HumanMessage):
        message_content = prompt_value.messages[0].content

    # Construct history for Cohere API (list of dicts with role/message)
    chat_history_list_for_api = [] # Corrected indentation
    if history_messages: # Use history from state # Corrected indentation
        for msg in history_messages:
            content = None
            role = None
            if isinstance(msg, HumanMessage):
                role = "USER"
                content = msg.content
            elif isinstance(msg, AIMessage): # Check for AIMessage
                role = "CHATBOT"
                content = msg.content
            elif isinstance(msg, dict): # Handle dict if necessary
                role = "USER" if msg.get("role") == "user" else "CHATBOT"
                content = msg.get("content")

            if role and content: # Ensure both role and content were found
                chat_history_list_for_api.append({"role": role, "message": content})
            else:
                 print(f"Warning: Skipping message with unexpected format in base_llm_node (Cohere path): {msg}")

    api_payload = {
        "message": message_content,
        "preamble": fallback_preamble.split("=== CONVERSATION HISTORY ===")[0].strip() + "\n\n" + fallback_preamble.split("=== INSTRUCTIONS ===")[1].strip(),
        "chat_history": chat_history_list_for_api, # Use the list of dicts for payload
        "model": fallback_llm.model_name,
        "temperature": fallback_llm.temperature,
    }
    prompt_composition_json_str = json.dumps(api_payload, indent=2)
    source_path = "base_llm_cohere"
    # Invoke Cohere chain (doesn't explicitly need chat_history in invoke dict if handled by prompt/binding)
    generation = llm_chain.invoke(fallback_input, config=config)

    # Return the exact prompt composition JSON string and source path
    return {"question": question, "generation": generation, "documents": [], "prompt_composition": prompt_composition_json_str, "source_path": source_path}

# Global variable for fallback preamble (Cohere)
fallback_preamble = get_fallback_preamble()

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
    doc_texts = "\n\n".join([doc.page_content for doc in documents if isinstance(doc, Document)])

    # Print formatted documents for debugging
    print(f"--- Formatted {len(formatted_docs)} documents for RAG ---")

    # Construct the prompt messages that will be sent
    rag_input = {"documents": documents, "question": question}
    prompt_value = rag_prompt(rag_input)

    # Construct the likely API payload for documentation
    global rag_preamble, rag_llm # Use Cohere globals
    prompt_value = rag_prompt(rag_input)
    # Extract message content
    message_content = ""
    if prompt_value.messages and isinstance(prompt_value.messages[0], HumanMessage):
        message_content = prompt_value.messages[0].content

    # Construct history for Cohere API (list of dicts with role/message)
    chat_history_list_for_api = [] # Corrected indentation
    if history_messages: # Use history from state # Corrected indentation
        for msg in history_messages:
            content = None
            role = None
            if isinstance(msg, HumanMessage):
                role = "USER"
                content = msg.content
            elif isinstance(msg, AIMessage): # Check for AIMessage
                role = "CHATBOT"
                content = msg.content
            elif isinstance(msg, dict): # Handle dict if necessary
                role = "USER" if msg.get("role") == "user" else "CHATBOT"
                content = msg.get("content")

            if role and content: # Ensure both role and content were found
                chat_history_list_for_api.append({"role": role, "message": content})
            else:
                print(f"Warning: Skipping message with unexpected format in generate_node: {msg}")

    # Format documents for Cohere API (list of dicts with 'text' key)
    cohere_docs_format = [{"text": doc.page_content if doc.page_content is not None else ""} for doc in documents if isinstance(doc, Document)] # Added None check

    api_payload = {
        "message": message_content,
        "preamble": rag_preamble.split("=== CONVERSATION HISTORY ===")[0].strip() + "\n\n" + rag_preamble.split("=== INSTRUCTIONS ===")[1].strip(), # Combine parts without history
        "chat_history": chat_history_list_for_api, # Use the list of dicts for payload
        "documents": cohere_docs_format, # Use Cohere-specific format
        "model": rag_llm.model_name,
        "temperature": rag_llm.temperature,
        # Other potential parameters like connectors, etc. are omitted for simplicity
    }
    prompt_composition_json_str = json.dumps(api_payload, indent=2)


    print("--- Invoking RAG chain ---")
    generation = "Error: RAG chain invocation failed." # Default error message
    try:
        # RAG chain invocation doesn't explicitly need chat_history if handled by prompt/binding
        generation = rag_chain.invoke(rag_input, config=config)
        print(f"--- Generation result: {generation[:100]}... ---")
    except Exception as e:
        print(f"!!! ERROR invoking Cohere RAG chain: {e} !!!")
        import traceback
        traceback.print_exc()
        # Keep the default error message

    # Return the exact prompt composition JSON string
    return {
        "documents": documents, # Keep original LangChain documents in state
        "question": question,
        "generation": generation,
        "prompt_composition": prompt_composition_json_str, # Use the correct variable name
        "generation_attempts": generation_attempts,
        "source_path": state.get("source_path"), # Preserve source_path from input state
        "history_messages": history_messages # Pass history messages through state
    }

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
    # Pass history_messages through
    return {"documents": filtered_docs, "question": question, "history_messages": state.get("history_messages")}

def web_search_node(state: GraphState):
    """Web search based on the question using Serper.dev."""
    print("---WEB SEARCH (Serper)---") # Updated print statement
    question = state["question"]
    config = state.get("config")

    if not SERPER_API_KEY: # Use correct variable
        print("---WEB SEARCH: SERPER_API_KEY not set. Skipping.---")
        return {"documents": [], "question": question, "source_path": "web_search_error"} # Indicate error

    # Perform web search using the updated function
    search_results = serper_search(question, SERPER_API_KEY) # Use correct variable and function

    # Format results into LangChain Documents
    web_docs = []
    if search_results:
        for result in search_results:
            # Adjust keys based on actual SerpDev response structure
            content = result.get("snippet", result.get("title", ""))
            metadata = {"source": "web_search", "url": result.get("link", "")}
            if content:
                web_docs.append(Document(page_content=content, metadata=metadata))

    if not web_docs:
         print("---WEB SEARCH: No results found.---")

    # Return documents as a list of LangChain Documents and pass history_messages
    return {"documents": web_docs, "question": question, "source_path": "web_search", "history_messages": state.get("history_messages")} # Set source_path


# --- Graph Edges ---

def route_question_edge(state: GraphState):
    """Route question to web search, vectorstore, or LLM fallback."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    config = state.get("config")
    
    # Check if Qdrant is available
    if vectorstore is None and retriever is None:
        print("---ROUTE: Qdrant not available, checking if web search is possible---")
        if SERPER_API_KEY: # Use correct variable
            print("---ROUTE: Using web search as Qdrant is not available---")
            return "web_search"
        else:
            print("---ROUTE: Using Base LLM as neither Qdrant nor web search are available---") # Updated print
            return "base_llm" # Route to renamed node
    
    # If Qdrant is available, use the router to decide
    source = question_router.invoke({"question": question}, config=config)

    # If the router doesn't choose a specific tool, default to BaseLLMTool
    if not source.tool_calls:
        print("---ROUTE: No specific tool called by router, using Base LLM---")
        return "base_llm" # Default to base_llm

    # Always take the first tool call if multiple are suggested
    datasource = source.tool_calls[0]["name"]

    if datasource == "WebSearchTool": # Use correct tool name
        print("---ROUTE: Question to Web Search (Serper)---") # Updated print
        return "web_search"
    elif datasource == "VectorstoreTool":
        print("---ROUTE: Question to Vectorstore---")
        return "vectorstore"
    elif datasource == "BaseLLMTool": # Handle BaseLLMTool
        print("---ROUTE: Question to Base LLM---")
        return "base_llm"
    else:
        # Should not happen with the defined tools, but fallback just in case
        print(f"---ROUTE: Unknown tool '{datasource}', using Base LLM as fallback---") # Updated print
        return "base_llm" # Fallback to base_llm

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
        "vectorstore": "retrieve",
        "base_llm": "base_llm", # Match the return value from route_question_edge
    },
)

# After retrieving from vectorstore, grade the documents
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
def run_adaptive_rag(question: str, coach_name: str = "AI Coach", persona: str = "", conversation_history: str = "", history_messages: Optional[List[Any]] = None, config: Optional[dict] = None) -> dict:
    """
    Runs the adaptive RAG graph for a given question.

    Args:
        question: The user's question.
        coach_name: Name of the coach.
        persona: Coach persona description.
        conversation_history: The conversation history formatted as a string (for preamble).
        history_messages: The raw list of BaseMessage objects (k=5) for accurate prompt composition.
        config: Optional LangSmith configuration dictionary.

    Returns:
        A dictionary containing the final 'generation', 'documents', 'prompt_composition', and 'source_path'.
    """
    if not config:
        config = {} # Default empty config if none provided

    # Update the LLM preambles with the coach name and persona
    global rag_llm, fallback_llm, rag_preamble, fallback_preamble
    rag_preamble = get_rag_preamble(coach_name, persona, conversation_history)
    fallback_preamble = get_fallback_preamble(coach_name, persona, conversation_history)

    # Rebind the LLMs with the updated preambles
    rag_llm = ChatCohere(model_name="command-r", temperature=0, api_key=COHERE_API_KEY).bind(
        preamble=rag_preamble
    )
    fallback_llm = ChatCohere(model_name="command-r", temperature=0, api_key=COHERE_API_KEY).bind(
        preamble=fallback_preamble
    )

    # Recreate the chains with the updated LLMs
    global rag_chain, llm_chain
    rag_chain = rag_prompt | rag_llm | StrOutputParser()
    llm_chain = fallback_prompt | fallback_llm | StrOutputParser()

    # Include history_messages in the initial graph input state
    # Include history_messages and potentially coach_name/persona if needed by nodes directly
    inputs = {
        "question": question,
        "config": config,
        "history_messages": history_messages or [],
        # Add coach_name and persona to state if generate_node needs them directly
        # "coach_name": coach_name,
        # "persona": persona,
    }
    # Removed pre-cleaning logic here; cleaning will happen inside nodes if needed.

    final_state = None
    print(f"\n--- Running Adaptive RAG for: '{question}' ---")
    try:
        # Increase recursion limit to handle complex questions
        stream_config = config.copy() if config else {}
        stream_config["recursion_limit"] = 50  # Increase from default 25
        
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
        # Set a minimal error state to ensure something is returned
        final_state = {
            "question": question,
            "generation": f"Error during graph execution: {e}",
            "documents": [],
            "source_path": "graph_error",
            "prompt_composition": None
        }

    # Extract the final generation
    # Extract the final generation and documents
    final_generation = "Sorry, I encountered an issue and couldn't generate a response."
    final_documents = []
    final_prompt_composition = None # Initialize prompt composition
    if final_state:
        final_generation = final_state.get("generation", final_generation)
        final_documents = final_state.get("documents", [])
        final_prompt_composition = final_state.get("prompt_composition") # Get prompt composition from final state
        
        # If we're using the fallback LLM but have documents, include them in the prompt composition
        if final_prompt_composition and "Fallback Prompt" in final_prompt_composition and final_documents:
            doc_texts = "\n\n".join([doc.page_content for doc in final_documents if isinstance(doc, Document)])
            # Add the documents to the prompt composition
            final_prompt_composition = final_prompt_composition.replace(
                "--- Fallback Prompt ---", 
                f"""--- RAG Prompt (Fallback Used) ---

=== Documents / Search results ===
{doc_texts}"""
            )
        
        # If prompt_composition doesn't include the persona, add it
        if final_prompt_composition and "COACH PERSONA" not in final_prompt_composition:
            source = "RAG" if final_documents else "Base LLM" # Updated source name
            final_prompt_composition = f"""=== COACH PERSONA ===
You are {coach_name} {persona}. Answer the user's question as their AI coach.

=== INSTRUCTIONS ===
- Always respond in the tone and style of {coach_name}, with a friendly and professional tone.
- Keep answers concise unless more detail is asked, and focus on actionable advice.
- NEVER mention that you are an AI language model or that you're using any specific information sources.

{final_prompt_composition}"""
        
        print("--- Adaptive RAG Finished ---")
    else:
         print("--- Adaptive RAG Finished (No final state found) ---")

    final_source_path = final_state.get("source_path") if final_state else None

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
    result_web = run_adaptive_rag(test_question_web)
    print(f"\nFinal Answer (Web): {result_web['generation']}")
    # print(f"Documents (Web): {result_web['documents']}")

    print("\nTesting RAG Route...")
    result_rag = run_adaptive_rag(test_question_rag)
    print(f"\nFinal Answer (RAG): {result_rag['generation']}")
    # print(f"Documents (RAG): {result_rag['documents']}")

    print("\nTesting Base LLM Route...") # Updated print
    result_base_llm = run_adaptive_rag(test_question_base_llm) # Use renamed variable
    print(f"\nFinal Answer (Base LLM): {result_base_llm['generation']}") # Updated print
