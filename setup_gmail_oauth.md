# Gmail OAuth Setup Instructions

## Step 1: Create New OAuth Credentials

1. **Open Google Cloud Console**: https://console.cloud.google.com/
2. **Select Project**: Use project "intricate-will-465320-q0" or create a new one
3. **Enable Gmail API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click on it and press "Enable"

4. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "+ CREATE CREDENTIALS" > "OAuth 2.0 Client IDs"
   - If prompted, configure the OAuth consent screen:
     - Choose "External" user type
     - Fill in required fields (App name: "EmailTool", User support email, etc.)
     - Add your email to test users
   - For Application type, select "Desktop application"
   - Name: "EmailTool Desktop Client"
   - Click "Create"
   - Download the JSON file

## Step 2: Replace Credentials

1. Save the downloaded file as `credentials.json`
2. Replace the existing `/Users/idrisalo/emailtool-new/credentials.json`

## Step 3: Test Authentication

Run the backend service and it will automatically open a browser for OAuth:

```bash
cd /Users/idrisalo/emailtool-new
source /Users/idrisalo/emailnlp-env/bin/activate
python advanced_ai_service.py
```

## Step 4: Verify Email Sync

Once authenticated, test the email sync endpoint:

```bash
curl http://localhost:8000/sync-emails
```

## Current Status
- ✅ Python backend with Gmail API integration ready
- ✅ PostgreSQL database configured
- ✅ Frontend ready to display emails
- ⏳ Need new OAuth credentials (current ones are deleted)
- ⏳ Need to complete Gmail authentication

## Troubleshooting

If you encounter issues:
1. Make sure the Gmail API is enabled in your Google Cloud project
2. Ensure your email is added as a test user in the OAuth consent screen
3. Check that the credentials.json file is properly formatted
4. Verify PostgreSQL is running: `brew services list | grep postgresql`
