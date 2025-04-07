# Entity Memories Directory

This directory stores entity memories for the AI Coach application. Each conversation's entity memory is saved as a JSON file with the conversation ID as the filename.

## File Format

Each entity memory file contains a JSON object where keys are entity names and values are entity information objects with the following structure:

```json
{
  "Product Market Fit": {
    "type": "BUSINESS_CONCEPT",
    "first_mentioned": "2025-04-06 19:15:22",
    "last_mentioned": "2025-04-06 19:20:45",
    "mentions": [
      "User asked about achieving product market fit for SaaS products",
      "You explained that product market fit means creating a product that satisfies a strong market demand"
    ],
    "summary": "The state where a product satisfies a strong market demand; critical for startup success",
    "attributes": {
      "definition": "When a product satisfies a strong market demand",
      "importance": "Critical milestone for startup success",
      "indicators": ["Customer retention", "Word-of-mouth growth", "Willing to pay"]
    }
  },
  "John's Startup": {
    "type": "ORGANIZATION",
    "first_mentioned": "2025-04-06 19:15:22",
    "last_mentioned": "2025-04-06 19:20:45",
    "mentions": [
      "User mentioned working on a B2B SaaS startup called 'John's Startup'",
      "User said John's Startup is targeting small accounting firms"
    ],
    "summary": "User's B2B SaaS startup targeting small accounting firms",
    "attributes": {
      "industry": "SaaS",
      "target_market": "Small accounting firms",
      "stage": "Early-stage"
    }
  }
}
```

## Usage

The entity memory is automatically updated when a user interacts with the AI Coach. Entities are extracted from both user messages and AI responses, and their information is stored and updated over time.

When generating responses, the AI Coach retrieves relevant entities based on the user's query and includes them in the context, allowing it to reference specific details about people, organizations, concepts, etc. that were mentioned in previous conversations.

## Implementation

Entity extraction uses a hybrid approach combining:

1. Rule-based extraction for common coaching entities
2. spaCy-based extraction for named entities (people, organizations, etc.)
3. LLM-based extraction and enrichment for more complex entities and attributes

This approach balances efficiency, accuracy, and comprehensiveness.
