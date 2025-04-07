# Serper.dev Setup Guide

The AI Coach app uses Serper.dev to provide web search functionality for time-sensitive questions. This guide explains how to set up and troubleshoot the Serper.dev integration.

## Current Status

**Web search functionality is now using Serper.dev API.**

The application will use Serper.dev to search for current information when answering time-sensitive questions. If the search fails for any reason, the app will fall back to using the base language model and include a note in the response indicating that the answer was generated without access to current information.

## About Serper.dev

[Serper.dev](https://serper.dev) is a Google Search API that provides real-time search results in JSON format. It's a reliable alternative to SerpAPI and offers similar functionality.

## API Key Configuration

The application is currently configured to use your Serper.dev API key. The key is stored in the `.env` file under the `SERPAPI_API_KEY` variable (we're reusing the same environment variable for simplicity).

## Updating Your API Key

If you need to update your API key:

1. Open the `.env` file in the root directory of the AI Coach app
2. Find the line that starts with `SERPAPI_API_KEY=`
3. Replace the existing key with your new key:
   ```
   SERPAPI_API_KEY=your-new-key-here
   ```
4. Save the file
5. Restart the application for the changes to take effect

## Troubleshooting

If web search is not working, check the following:

### 1. Verify Your API Key

Make sure your API key is valid and active:
- Log in to your [Serper.dev dashboard](https://serper.dev/dashboard)
- Check that your account is active and has available credits
- Verify that the API key in your `.env` file matches the one in your dashboard

### 2. Check API Usage Limits

Serper.dev has usage limits depending on your plan:
- Free tier: Limited number of searches per month
- If you've exceeded your limit, you'll need to upgrade your plan or wait until the next billing cycle

### 3. Check the Application Logs

The application logs detailed information about web search attempts:
- Check the `app_debug.log` file in the root directory
- Look for lines containing "Serper.dev" to see specific error messages

### 4. Network Issues

If your network has restrictions on outbound connections:
- Ensure that your network allows connections to google.serper.dev
- Check if you need to configure a proxy for external API calls

## Alternative Solutions

If you're unable to get Serper.dev working, consider these alternatives:

1. **Use a different search API**: The application can be modified to use alternative search APIs like Bing Search API or Google Custom Search API.

2. **Implement a simple web scraper**: For basic search functionality, a custom web scraper could be implemented.

3. **Disable web search routing**: Configure the application to always use the base LLM or RAG system, avoiding the need for web search.

## Support

If you continue to experience issues with the web search functionality, please:

1. Check the application logs for specific error messages
2. Contact SerpAPI support if the issue is related to your API key or account
3. Consider implementing one of the alternative solutions mentioned above
