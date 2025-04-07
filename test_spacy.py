import spacy

# Load the spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully!")

# Test text
text = "Apple is looking at buying U.K. startup for $1 billion. John Smith is the CEO of the company."
print(f"\nAnalyzing text: '{text}'")

# Process the text
doc = nlp(text)

# Print entities
print("\nEntities found:")
for ent in doc.ents:
    print(f"  - {ent.text} ({ent.label_})")

# Print tokens and their parts of speech
print("\nTokens and parts of speech:")
for token in doc[:10]:  # Just show the first 10 tokens
    print(f"  - {token.text} ({token.pos_})")

