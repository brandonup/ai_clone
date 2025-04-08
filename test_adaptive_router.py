from utils.adaptive_router import run_adaptive_rag

# Test the adaptive RAG function
def test_adaptive_rag():
    # Test question about Paul Graham and co-founders
    question = "What does Paul Graham say about startups and co-founders?"
    coach_name = "AI Coach"
    persona = "an expert in startups and entrepreneurship"
    
    print(f"Testing question: {question}")
    result = run_adaptive_rag(
        question=question,
        coach_name=coach_name,
        persona=persona,
        conversation_history=""
    )
    
    # Print the results
    print("\n=== RESULTS ===")
    print(f"Generation: {result.get('generation')}")
    print(f"\nNumber of documents: {len(result.get('documents', []))}")
    
    # Print document content
    documents = result.get('documents', [])
    if documents:
        print("\n=== DOCUMENT CONTENT ===")
        for i, doc in enumerate(documents):
            print(f"Document {i+1}:")
            print(doc.page_content)
            print("-" * 50)
    
    # Print prompt composition
    prompt_composition = result.get('prompt_composition')
    if prompt_composition:
        print("\n=== PROMPT COMPOSITION ===")
        print(prompt_composition)

    # Print source path
    source_path = result.get('source_path')
    print(f"\nSource Path: {source_path}")

if __name__ == "__main__":
    # Test only the web search question
    question = "What is the weather like in San Francisco today?"
    coach_name = "AI Coach"
    persona = "an expert in startups and entrepreneurship"
    
    print(f"Testing question: {question}")
    result = run_adaptive_rag(
        question=question,
        coach_name=coach_name,
        persona=persona,
        conversation_history=""
    )
    
    # Print the results
    print("\n=== RESULTS ===")
    print(f"Generation: {result.get('generation')}")
    print(f"\nNumber of documents: {len(result.get('documents', []))}")
    
    # Print document content
    documents = result.get('documents', [])
    if documents:
        print("\n=== DOCUMENT CONTENT ===")
        for i, doc in enumerate(documents):
            print(f"Document {i+1}:")
            print(doc.page_content)
            print("-" * 50)
    
    # Print prompt composition
    prompt_composition = result.get('prompt_composition')
    if prompt_composition:
        print("\n=== PROMPT COMPOSITION ===")
        print(prompt_composition)

    # Print source path
    source_path = result.get('source_path')
    print(f"\nSource Path: {source_path}")
