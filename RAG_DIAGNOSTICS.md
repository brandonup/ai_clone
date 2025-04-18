# RAG Diagnostics Guide

This guide explains how to diagnose and fix issues with the Retrieval-Augmented Generation (RAG) system in the AI Clone application.

## Overview

The RAG system in this application uses Ragie.ai for document retrieval and combines it with various LLM providers (OpenAI, Cohere, Google) to generate responses. When the RAG system isn't working properly, it could be due to several issues:

1. API key problems (expired, invalid, or missing)
2. Rate limiting on one or more API services
3. Network connectivity issues
4. Document ingestion problems
5. Routing configuration issues

## Using the Diagnostic Tools

Several diagnostic tools are provided to help identify and fix RAG issues:

### 1. diagnose_rag.sh

This is a shell script that runs the Python diagnostic tool and displays the results.

To use it:

```bash
# Make the script executable
chmod +x diagnose_rag.sh

# Run the script
./diagnose_rag.sh
```

### 2. diagnose_rag.py

This is a Python script that performs detailed diagnostics on the RAG system. It checks:

- API key validity for all services
- Document retrieval from Ragie.ai
- Adaptive router functionality
- Rate limit issues
- Automatically pulls clone data from Firestore for more accurate testing

You can run it directly with:

```bash
python3 diagnose_rag.py
```

The script will create a detailed log file (`rag_diagnostics.log`) with the results.

## Interpreting the Results

The diagnostic tools will provide a summary of the tests performed and their results. Here's how to interpret them:

### API Key Validation

If any API keys are invalid or missing, you'll see error messages like:

```
❌ FAIL - api_keys_valid
```

**Fix**: Check the `.env` file and ensure all required API keys are set and valid.

### Ragie Retrieval

If document retrieval from Ragie.ai is failing, you'll see:

```
❌ FAIL - ragie_retrieval_working
```

**Fix**:
- Verify your Ragie API key
- Check if documents were properly ingested
- Check the Ragie.ai service status
- Ensure your collection configuration is correct

### Adaptive Router

If the adaptive router is not working correctly, you'll see:

```
❌ FAIL - adaptive_router_working
```

**Fix**:
- Review the `adaptive_router.py` implementation
- Check if the router is correctly configured to use Ragie
- Verify that the router can properly route to different LLMs

### Rate Limit Issues

If any API is hitting rate limits, you'll see:

```
❌ FAIL - no_rate_limit_issues
```

**Fix**:
- Implement rate limiting or backoff strategies
- Check your API usage and consider upgrading your plan
- Distribute requests more evenly over time

## Common Issues and Solutions

### 1. No Documents Retrieved

If the system isn't retrieving any documents from Ragie.ai:

- Check if documents were properly ingested
- Verify the collection name and partition configuration
- Test with simple, relevant queries
- Check the Ragie API key and service status

### 2. Rate Limiting

If you're hitting rate limits:

- Implement exponential backoff for retries
- Cache responses where appropriate
- Reduce the frequency of requests
- Upgrade your API plan if necessary

### 3. Routing Issues

If queries are being routed incorrectly:

- Check the router configuration
- Verify that the router can access all necessary services
- Test with different types of queries
- Review the routing logic in `adaptive_router.py`

### 4. API Key Issues

If API keys are invalid or expired:

- Generate new API keys
- Update the `.env` file
- Restart the application
- Verify key permissions and scopes

## Advanced Troubleshooting

For more advanced troubleshooting:

1. Check the application logs for specific error messages
2. Use the LangSmith tracing (if configured) to analyze the execution flow
3. Test individual components separately (e.g., just Ragie retrieval, just LLM generation)
4. Monitor API usage and response times
5. Check network connectivity and firewall settings

## Getting Help

If you're still experiencing issues after running the diagnostics and applying the suggested fixes, consider:

1. Checking the Ragie.ai documentation and status page
2. Reviewing the documentation for the LLM providers you're using
3. Checking for updates to the libraries and dependencies
4. Reaching out to the respective support teams

## Maintaining the RAG System

To keep your RAG system running smoothly:

1. Regularly update your API keys
2. Monitor your API usage and costs
3. Keep your document collections up to date
4. Test the system periodically with the diagnostic tools
5. Keep dependencies updated
