# Custom RAG Implementation

This document explains the custom RAG (Retrieval Augmented Generation) implementation with Small-to-Big chunking and sliding window strategies.

## Overview

The custom RAG system enhances the AI Coach's ability to retrieve and use relevant information from your documents by implementing two advanced chunking strategies:

1. **Small-to-Big Chunking**: Uses small chunks (175 tokens) for retrieval and their parent large chunks (512 tokens) for context generation
2. **Sliding Window**: Creates overlapping chunks to ensure information isn't lost at chunk boundaries

## How It Works

### Small-to-Big Chunking

Traditional RAG systems face a dilemma:
- Small chunks are better for precise retrieval
- Large chunks provide better context for the LLM

Our Small-to-Big approach solves this by:
1. Creating large chunks (512 tokens) from your documents
2. Breaking each large chunk into smaller chunks (175 tokens)
3. Using the small chunks for vector search (retrieval)
4. Sending the parent large chunks to the LLM (generation)

This gives you the precision of small-chunk retrieval with the context of large-chunk generation.

### Sliding Window

To ensure information isn't lost at chunk boundaries, we implement a sliding window approach:
- Large chunks overlap by 100 tokens
- Small chunks overlap by 50 tokens

This ensures that concepts or information that would normally be split across chunks are fully captured in at least one chunk.

## Technical Implementation

The custom RAG system uses:
- **Qdrant**: A vector database for storing and retrieving embeddings
- **OpenAI Embeddings**: For generating high-quality vector representations
- **Tiktoken**: For accurate token counting and chunking

## Configuration

The custom RAG implementation can be enabled or disabled using the `.env` file:

```
# Custom RAG Configuration
USE_CUSTOM_RAG=true
```

Set to `true` to use the custom RAG implementation, or `false` to use the original Ragie.ai implementation.

## Benefits

1. **Improved Retrieval Precision**: Small chunks excel at finding specific information
2. **Better Context Understanding**: Large chunks provide the LLM with more context
3. **No Information Loss**: Sliding window ensures information isn't lost at chunk boundaries
4. **Fallback Mechanism**: Falls back to keyword search if vector search returns no results
5. **Rate Limit Handling**: Processes embeddings in batches with automatic retries to avoid OpenAI rate limits
6. **Large Document Support**: Automatically splits very large documents into manageable sections for processing

## Google Drive Integration

The Google Drive integration is now completely optional. You can:
- Provide a Google Drive folder URL to use your own documents
- Leave the field empty to use the AI Coach without any custom documents

## Reverting to Original Implementation

If you encounter any issues with the custom RAG implementation, you can revert to the original Ragie.ai implementation by setting `USE_CUSTOM_RAG=false` in the `.env` file.
