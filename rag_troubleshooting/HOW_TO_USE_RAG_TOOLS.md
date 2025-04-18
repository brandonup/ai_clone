# How to Use the RAG Diagnostic and Fixing Tools

This guide explains how to use the tools provided to diagnose and fix issues with the Retrieval-Augmented Generation (RAG) system in the AI Clone application.

## Quick Start

1. Run the diagnostic tool to identify issues:
   ```bash
   ./diagnose_rag.sh
   ```
   The tool will automatically pull clone data from Firestore for more accurate testing.

2. Review the diagnostic results in the terminal and in `rag_diagnostics.log`

3. Fix identified issues using the fixer tool:
   ```bash
   ./fix_rag_issues.sh
   ```

4. Follow the interactive prompts to fix specific issues

## Detailed Usage

### Diagnosing RAG Issues

The diagnostic tool checks for several common issues:

1. **API Key Validation**: Verifies that all required API keys are set and valid
2. **Ragie Retrieval**: Tests document retrieval from Ragie.ai
3. **Adaptive Router**: Tests the routing functionality with different query types
4. **Rate Limit Issues**: Checks if any APIs are hitting rate limits

To run the diagnostic tool:

```bash
./diagnose_rag.sh
```

This will execute the Python script `diagnose_rag.py` and display the results in the terminal. A detailed log is also saved to `rag_diagnostics.log`.

You can also run the Python script directly with:

```bash
python3 diagnose_rag.py
```

### Fixing RAG Issues

The fixer tool provides an interactive interface to fix identified issues:

1. **API Key Issues**: Validates and updates API keys
2. **Ragie Retrieval Issues**: Tests and fixes document retrieval
3. **Adaptive Router Issues**: Tests and fixes the routing functionality
4. **Rate Limit Issues**: Provides guidance on implementing rate limiting strategies

To run the fixer tool:

```bash
./fix_rag_issues.sh
```

This will display a menu of options to fix specific issues. Select the appropriate option based on the diagnostic results.

You can also run the Python script directly with specific options:

```bash
# Fix API key issues
python3 fix_rag_issues.py --api-keys

# Fix Ragie retrieval issues
python3 fix_rag_issues.py --ragie

# Fix adaptive router issues
python3 fix_rag_issues.py --router

# Fix rate limit issues
python3 fix_rag_issues.py --rate-limits

# Fix all issues
python3 fix_rag_issues.py --all
```

## Common Issues and Solutions

### 1. Invalid or Expired API Keys

**Symptoms**:
- "API key is invalid" errors in logs
- Authentication failures
- Empty responses from API calls

**Solution**:
1. Run `./fix_rag_issues.sh` and select option 1 (API key issues)
2. Follow the prompts to update the invalid API keys
3. The tool will validate the new keys and update the `.env` file

### 2. No Documents Retrieved

**Symptoms**:
- Empty results from Ragie.ai
- "No documents found" messages
- RAG responses falling back to base LLM

**Solution**:
1. Run `./fix_rag_issues.sh` and select option 2 (Ragie retrieval issues)
2. The tool will check if documents are properly ingested
3. You may be prompted to reset the document store if necessary

### 3. Incorrect Routing

**Symptoms**:
- Queries being routed to the wrong source (e.g., web search instead of vectorstore)
- Inconsistent responses to similar queries
- Unexpected fallbacks to base LLM

**Solution**:
1. Run `./fix_rag_issues.sh` and select option 3 (Adaptive router issues)
2. The tool will test the router with different query types
3. Follow any recommendations to fix routing configuration

### 4. Rate Limit Errors

**Symptoms**:
- "Rate limit exceeded" errors
- Temporary failures that resolve after waiting
- Inconsistent API responses during high usage

**Solution**:
1. Run `./fix_rag_issues.sh` and select option 4 (Rate limit issues)
2. The tool will provide guidance on implementing rate limiting strategies
3. Follow the code examples to add exponential backoff to API calls

## Maintenance Recommendations

To keep your RAG system running smoothly:

1. Run the diagnostic tool periodically to catch issues early
2. Update API keys before they expire
3. Monitor API usage and costs
4. Keep document collections up to date
5. Implement proper error handling and rate limiting in production code

## Additional Resources

For more detailed information, refer to:

- [RAG_DIAGNOSTICS.md](RAG_DIAGNOSTICS.md) - Comprehensive guide to diagnosing and fixing RAG issues
- [Ragie.ai Documentation](https://docs.ragie.ai) - Official documentation for the Ragie.ai service
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) - Documentation for the LangChain framework
