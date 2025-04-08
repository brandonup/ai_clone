import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_cohere import ChatCohere

# Load environment variables
load_dotenv()

# Get API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def test_cohere_rag():
    # Initialize Cohere LLM
    llm = ChatCohere(model="command-r", temperature=0, api_key=COHERE_API_KEY)
    
    # Create a test document about Paul Graham and co-founders
    documents = [
        Document(
            page_content="105 Startups in 13 Sentences Want to start a startup? Get funded by Y Combinator. \nWatch how this essay was written. \nFebruary 2009 One of the things I always tell startups is a principle I learned from Paul Buchheit: it's better to make a few people really happy than to make a lot of people semi-happy. \nI was saying recently to a reporter that if I could only tell startups 10 things this would be one of them. \nThen I thought: what would the other 9 be? When I made the list there turned out to be 13: 1\\. \nPick good cofounders. \nCofounders are for a startup what location is for real estate. \nYou can change anything about a house except where it is. \nIn a startup you can change your idea easily but changing your cofounders is hard.",
            metadata={"source": "Paul Graham's essay"}
        )
    ]
    
    # Format documents for Cohere
    formatted_docs = [{"text": doc.page_content} for doc in documents]
    
    # Test question
    question = "What does Paul Graham say about startups and co-founders?"
    
    # Create a prompt with the documents
    from langchain_core.messages import HumanMessage
    
    # Create a message with documents in Cohere's format
    message = HumanMessage(
        f"Question: {question} \nAnswer: ",
        additional_kwargs={"documents": formatted_docs}
    )
    
    # Get the response
    response = llm.invoke([message])
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Question: {question}")
    print(f"Response: {response.content}")
    
    # Print document content
    print("\n=== DOCUMENT CONTENT ===")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(doc.page_content)
        print("-" * 50)

if __name__ == "__main__":
    test_cohere_rag()
