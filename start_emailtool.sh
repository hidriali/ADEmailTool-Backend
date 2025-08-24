#!/bin/bash

# EmailTool - Gmail OAuth Setup and Service Restart Script
# Run this script after you've obtained new OAuth credentials

echo "🔧 EmailTool - Setting up Gmail OAuth and restarting services"
echo "================================================================"

# Check if new credentials.json exists
if [ ! -f "credentials.json" ]; then
    echo "❌ ERROR: credentials.json not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://console.cloud.google.com/"
    echo "2. Select project 'intricate-will-465320-q0' (or create new)"
    echo "3. Enable Gmail API in APIs & Services > Library"
    echo "4. Create OAuth 2.0 credentials in APIs & Services > Credentials"
    echo "5. Choose 'Desktop application' type"
    echo "6. Download and save as 'credentials.json' in this directory"
    echo ""
    echo "For detailed instructions, see: setup_gmail_oauth.md"
    exit 1
fi

echo "✅ Found credentials.json"

# Remove old token if exists
if [ -f "token.json" ]; then
    echo "🗑️  Removing old authentication token..."
    rm token.json
fi

# Test authentication
echo "🔐 Testing Gmail authentication..."
python test_gmail_auth.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Gmail authentication successful!"
    echo ""
    echo "🚀 Starting EmailTool backend service..."
    echo "The service will be available at http://localhost:8000"
    echo ""
    echo "API endpoints:"
    echo "- GET  /emails        - Get all emails from database"
    echo "- POST /sync-emails   - Sync emails from Gmail to database"
    echo "- POST /analyze-email - Analyze email content with AI"
    echo ""
    echo "Press Ctrl+C to stop the service"
    echo ""
    
    # Start the backend service
    python advanced_ai_service.py
else
    echo ""
    echo "❌ Gmail authentication failed!"
    echo "Please check the setup instructions and try again."
    exit 1
fi
