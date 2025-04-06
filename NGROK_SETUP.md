# Using Ngrok with AI Coach App

This guide explains how to use Ngrok to make your locally running AI Coach app accessible from any network.

## What is Ngrok?

Ngrok is a tool that creates secure tunnels to localhost, allowing external access to a locally running web server. This is useful for:

- Sharing your application with others without deploying it
- Testing webhooks
- Accessing your application from different devices or networks

## Current Ngrok URL

The AI Coach app is currently accessible at:

**https://9ca4-97-115-137-157.ngrok-free.app**

This URL forwards to your locally running Flask application on port 5005.

## How to Use

1. Make sure your Flask application is running on port 5005
2. Make sure the ngrok tunnel is active (run `ngrok http 5005` if it's not)
3. Share the ngrok URL with anyone who needs to access your application
4. They can use the application as if it were deployed on the internet

## Important Notes

- The free version of ngrok will generate a new URL each time you restart the tunnel
- The tunnel will close if you shut down your computer or terminate the ngrok process
- If you need to restart the tunnel, run `ngrok http 5005` again and update this document with the new URL

## Starting Ngrok

If the tunnel is not active, you can start it with:

```bash
ngrok http 5005
```

## Checking Tunnel Status

You can check the status of your ngrok tunnels by visiting:

http://localhost:4040

This will show you the ngrok web interface with information about your active tunnels and request logs.
