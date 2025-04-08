import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables from .env file
load_dotenv()

# Get the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
else:
    print("GOOGLE_API_KEY found.")
    
    # Define permissive safety settings for debugging
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    try:
        # Initialize the Gemini model
        print("Initializing ChatGoogleGenerativeAI...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-03-25", 
            google_api_key=google_api_key,
            safety_settings=safety_settings,
            temperature=0 # Keep temperature low for predictable test
        )
        print("Initialization successful.")
        
        # Define a simple test prompt
        test_prompt = "What is 2 + 2?"
        print(f"Invoking model with prompt: '{test_prompt}'")
        
        # Invoke the model
        response = llm.invoke(test_prompt)
        
        # Print the result
        print("\n--- Gemini Response ---")
        print(response)
        if hasattr(response, 'content'):
             print(f"\nContent: {response.content}")
        print("----------------------")
        
    except Exception as e:
        print(f"\n!!! An error occurred during Gemini test: {e} !!!")
        import traceback
        traceback.print_exc()
