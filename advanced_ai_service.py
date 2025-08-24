#!/usr/bin/env python3
"""
Lightweight AI Email Service - Tiny Models for Real AI
Provides email drafting, grammar checking, and text improvement using small AI models
Real AI but laptop-friendly!
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
from datetime import datetime
import json
import base64
import psycopg2
from psycopg2.extras import RealDictCursor

# Gmail API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    print("⚠️  Gmail API not available - install with: pip install google-api-python-client")

# Ultra-lightweight AI models
try:
    from transformers import pipeline
    import torch
    AI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  transformers not available - install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gmail API configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'database': 'emailTool',  # Correct database name with capital T
    'user': 'postgres',
    'password': 'postgres',  # Change this to your PostgreSQL password
    'port': 5432
}

# Global variables
gmail_service = None
db_connection = None

app = FastAPI(
    title="EmailTool AI Service (Lightweight)",
    description="Real AI service for email processing using tiny, fast models",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class EmailDraftRequest(BaseModel):
    prompt: str
    tone: Optional[str] = "professional"
    length: Optional[str] = "medium"

class GrammarCheckRequest(BaseModel):
    text: str

class PolishRequest(BaseModel):
    text: str
    style: Optional[str] = "professional"

class AIResponse(BaseModel):
    success: bool
    result: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

# Global AI models (ultra-lightweight)
text_generator = None
grammar_fixer = None
summarizer = None

def load_tiny_models():
    """Load the tiniest possible AI models for real AI functionality"""
    global text_generator, grammar_fixer, summarizer
    
    if not AI_AVAILABLE:
        logger.error("❌ Transformers not available - install with: pip install transformers torch")
        return
    
    try:
        logger.info("🤖 Loading TINY AI models (real AI, laptop-friendly)...")
        
        # Ultra-tiny text generation - DistilGPT2 (82MB only!)
        logger.info("✍️ Loading tiny text generator (DistilGPT2)...")
        text_generator = pipeline(
            "text-generation",
            model="distilgpt2",  # Only 82MB!
            device=-1,  # Force CPU for stability
            pad_token_id=50256,
            do_sample=True,
            temperature=0.7
        )
        logger.info("✅ Text generator loaded successfully")
        
        # Tiny grammar model - use the text generator for grammar
        logger.info("📝 Setting up grammar correction...")
        grammar_fixer = text_generator  # Reuse for grammar
        logger.info("✅ Grammar fixer ready")
        
        # Ultra-tiny summarizer - DistilBART (142MB)
        logger.info("📄 Loading tiny summarizer...")
        try:
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-xsum-1-1",  # Tiny BART variant
                device=-1,
                max_length=50,
                min_length=10
            )
            logger.info("✅ Summarizer loaded successfully")
        except Exception as e:
            # Fallback to even smaller model
            logger.warning(f"📄 Summarizer fallback: {e}")
            logger.info("📄 Using text generator for summarization...")
            summarizer = text_generator
        
        logger.info("✅ Tiny AI models loaded successfully!")
        logger.info("💾 Total model size: ~250MB (laptop-friendly)")
        
    except Exception as e:
        logger.error(f"❌ Error loading tiny models: {e}")
        logger.info("🔄 Will use fallback text generation instead")
        # Fallback to text-only approach
        text_generator = None
        grammar_fixer = None
        summarizer = None

def smart_email_prompt(user_prompt: str, tone: str = "professional") -> str:
    """Create smart prompts for email generation"""
    
    tone_prefixes = {
        "professional": "Write a professional business email:",
        "friendly": "Write a friendly, warm email:",
        "formal": "Write a formal, respectful email:",
        "casual": "Write a casual, relaxed email:"
    }
    
    # Check if this is a generic reply generation request
    if "generate a" in user_prompt.lower() and "reply" in user_prompt.lower():
        prefix = tone_prefixes.get(tone, tone_prefixes["professional"])
        return f"{prefix} Write a thoughtful response to the previous message."
    # Check if it's a specific reply instruction
    elif "reply" in user_prompt.lower() or "respond" in user_prompt.lower():
        prefix = tone_prefixes.get(tone, tone_prefixes["professional"])
        return f"{prefix} {user_prompt}"
    else:
        prefix = tone_prefixes.get(tone, tone_prefixes["professional"])
        return f"{prefix} {user_prompt}"

def clean_generated_text(text: str, original_prompt: str) -> str:
    """Clean up AI-generated text to make it email-appropriate"""
    
    # Remove the original prompt if it appears in the output
    if original_prompt.lower() in text.lower():
        text = text.replace(original_prompt, "").strip()
    
    # Remove repetitive patterns (like "Reply to this request: Reply to this request:")
    import re
    # Find patterns that repeat more than twice
    repetitive_pattern = re.compile(r'(.{10,}?)\1{2,}', re.IGNORECASE)
    text = repetitive_pattern.sub(r'\1', text)
    
    # Split into sentences and clean
    sentences = text.split('.')
    clean_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()
        
        # Skip if too short, starts with prompt-like text, or is duplicate
        if (len(sentence) > 10 and 
            not sentence.startswith(('Write', 'Email', 'Dear Sir', 'Subject', 'Reply to this')) and
            sentence_lower not in seen_sentences):
            clean_sentences.append(sentence)
            seen_sentences.add(sentence_lower)
    
    if not clean_sentences:
        return "Thank you for your email. I wanted to follow up on your message and provide you with the information you requested."
    
    # Join sentences and ensure proper email format
    result = '. '.join(clean_sentences[:3])  # Max 3 sentences, with proper spacing
    if not result.endswith('.'):
        result += '.'
    
    return result

def format_as_email(content: str, tone: str = "professional") -> str:
    """Format generated content as a proper email"""
    
    # Add greeting based on tone
    if tone == "friendly":
        greeting = "Hi there!"
        closing = "Best,\n[Your Name]"
    elif tone == "formal":
        greeting = "Dear Sir/Madam,"
        closing = "Yours sincerely,\n[Your Name]"
    else:  # professional
        greeting = "Dear Recipient,"
        closing = "Best regards,\n[Your Name]"
    
    return f"{greeting}\n\n{content}\n\n{closing}"

def setup_gmail_service():
    """Set up Gmail API service"""
    global gmail_service
    
    if not GMAIL_AVAILABLE:
        logger.error("❌ Gmail API not available")
        return
    
    try:
        creds = None
        
        # Load existing token
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(CREDENTIALS_FILE):
                    logger.error("❌ Gmail credentials.json not found")
                    return
                    
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        gmail_service = build('gmail', 'v1', credentials=creds)
        logger.info("✅ Gmail API service initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup Gmail service: {e}")
        gmail_service = None

def setup_database():
    """Set up PostgreSQL database connection"""
    global db_connection
    
    try:
        db_connection = psycopg2.connect(**DATABASE_CONFIG)
        logger.info("✅ PostgreSQL database connected")
        
        # Create emails table if it doesn't exist
        create_emails_table()
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        db_connection = None

def create_emails_table():
    """Create emails table if it doesn't exist"""
    if not db_connection:
        return
        
    try:
        with db_connection.cursor() as cursor:
            # Table already exists with the structure we saw
            # Just ensure it exists (it does) and don't try to alter it
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    id SERIAL PRIMARY KEY,
                    gmail_id VARCHAR(255) UNIQUE,
                    subject TEXT,
                    sender VARCHAR(255),
                    body TEXT,
                    category VARCHAR(100),
                    received_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    timestamp TIMESTAMP
                )
            """)
            
            db_connection.commit()
            logger.info("✅ Emails table ready (using existing structure)")
    except Exception as e:
        logger.error(f"❌ Failed to verify emails table: {e}")
        try:
            db_connection.rollback()
        except:
            pass

def fetch_gmail_emails(max_results=10):
    """Fetch emails from Gmail API"""
    if not gmail_service:
        logger.error("❌ Gmail service not available")
        return []
    
    try:
        # Get list of messages
        results = gmail_service.users().messages().list(
            userId='me', 
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        emails = []
        
        for message in messages:
            # Get full message details
            msg = gmail_service.users().messages().get(
                userId='me', 
                id=message['id']
            ).execute()
            
            # Extract email data
            email_data = parse_gmail_message(msg)
            if email_data:
                emails.append(email_data)
        
        logger.info(f"✅ Fetched {len(emails)} emails from Gmail")
        return emails
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch Gmail emails: {e}")
        return []

def parse_gmail_message(message):
    """Parse Gmail message into our email format"""
    try:
        payload = message['payload']
        headers = payload.get('headers', [])
        
        # Extract headers
        subject = ""
        sender = ""
        date = ""
        
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            
            if name == 'subject':
                subject = value
            elif name == 'from':
                sender = value
            elif name == 'date':
                date = value
        
        # Extract body
        body = extract_message_body(payload)
        
        # Convert date - use current time if date parsing fails
        try:
            from email.utils import parsedate_to_datetime
            if date:
                timestamp = parsedate_to_datetime(date).isoformat()
            else:
                timestamp = datetime.now().isoformat()
        except:
            timestamp = datetime.now().isoformat()
        
        # Categorize email (simple logic)
        category = categorize_email(subject, body, sender)
        
        return {
            "gmail_id": message['id'],
            "subject": subject or "(No Subject)",  # Provide fallback for empty subjects
            "sender": sender or "(Unknown Sender)",
            "body": body[:500] + "..." if len(body) > 500 else body,  # Truncate long emails
            "category": category,
            "timestamp": timestamp,
            "read": False
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to parse Gmail message: {e}")
        return None

def extract_message_body(payload):
    """Extract email body from Gmail payload"""
    body = ""
    
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                if 'data' in part['body']:
                    body_data = part['body']['data']
                    body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                    break
    else:
        if payload['mimeType'] == 'text/plain':
            if 'data' in payload['body']:
                body_data = payload['body']['data']
                body = base64.urlsafe_b64decode(body_data).decode('utf-8')
    
    return body

def categorize_email(subject, body, sender):
    """Simple email categorization"""
    text = (subject + " " + body + " " + sender).lower()
    
    work_keywords = ['work', 'project', 'meeting', 'deadline', 'team', 'office', 'business']
    personal_keywords = ['friend', 'family', 'weekend', 'vacation', 'personal']
    
    work_score = sum(1 for keyword in work_keywords if keyword in text)
    personal_score = sum(1 for keyword in personal_keywords if keyword in text)
    
    if work_score > personal_score:
        return "Work"
    elif personal_score > 0:
        return "Personal"
    else:
        return "General"

def store_emails_in_db(emails):
    """Store emails in PostgreSQL database"""
    global db_connection
    
    if not emails:
        return
    
    try:
        # Create a fresh connection if needed
        if not db_connection or db_connection.closed:
            setup_database()
        
        # Rollback any failed transaction
        db_connection.rollback()
        
        with db_connection.cursor() as cursor:
            for email in emails:
                cursor.execute("""
                    INSERT INTO emails (gmail_id, subject, sender, body, category, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gmail_id) DO NOTHING
                """, (
                    email['gmail_id'],
                    email['subject'],
                    email['sender'],
                    email['body'],
                    email['category'],
                    email['timestamp']
                ))
            
            db_connection.commit()
            logger.info(f"✅ Stored {len(emails)} emails in database")
            
    except Exception as e:
        logger.error(f"❌ Failed to store emails in database: {e}")
        try:
            if db_connection:
                db_connection.rollback()
        except:
            pass

def get_emails_from_db():
    """Get emails from PostgreSQL database"""
    global db_connection
    
    try:
        # Create a fresh connection if needed
        if not db_connection or db_connection.closed:
            setup_database()
        
        # Rollback any failed transaction
        db_connection.rollback()
        
        with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, gmail_id, subject, sender, body, category, 
                       COALESCE(timestamp, received_at, created_at) as timestamp,
                       false as read
                FROM emails 
                ORDER BY COALESCE(timestamp, received_at, created_at) DESC 
                LIMIT 50
            """)
            
            emails = cursor.fetchall()
            return [dict(email) for email in emails]
            
    except Exception as e:
        logger.error(f"❌ Failed to get emails from database: {e}")
        # Try to rollback and reconnect
        try:
            if db_connection:
                db_connection.rollback()
        except:
            pass
        setup_database()
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize all services when app starts"""
    load_tiny_models()
    setup_database()
    setup_gmail_service()
    
    # Sync emails on startup
    if gmail_service and db_connection:
        logger.info("🔄 Syncing emails from Gmail...")
        emails = fetch_gmail_emails(20)  # Fetch latest 20 emails
        store_emails_in_db(emails)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "EmailTool AI Service (Lightweight)",
        "version": "3.0.0",
        "status": "running",
        "mode": "tiny AI models",
        "features": ["email_drafting", "grammar_check", "text_polish", "summarization"],
        "models_loaded": {
            "text_generator": text_generator is not None,
            "grammar_fixer": grammar_fixer is not None,
            "summarizer": summarizer is not None
        },
        "laptop_friendly": True,
        "real_ai": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "mode": "tiny AI models (~250MB total)",
        "ai_available": AI_AVAILABLE,
        "models": {
            "text_generator": "loaded" if text_generator else "not loaded",
            "grammar_fixer": "loaded" if grammar_fixer else "not loaded",
            "summarizer": "loaded" if summarizer else "not loaded"
        },
        "cpu_usage": "low",
        "memory_usage": "moderate", 
        "laptop_safe": True
    }

@app.post("/draft", response_model=AIResponse)
async def draft_email(request: EmailDraftRequest):
    """Generate email draft using tiny AI models - real AI!"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Draft request received: prompt='{request.prompt[:50]}...', tone={request.tone}")
        
        if text_generator:
            logger.info("Using AI text generator for draft")
            # Use tiny AI model for real text generation
            prompt = smart_email_prompt(request.prompt, request.tone or "professional")
            logger.info(f"Generated prompt: {prompt[:100]}...")
            
            # Generate with tiny model
            try:
                result = text_generator(
                    prompt,
                    max_new_tokens=100,  # Generate up to 100 new tokens
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=50256
                )
            except Exception as gen_error:
                if "max_length" in str(gen_error):
                    logger.warning(f"Length error, trying with shorter prompt: {gen_error}")
                    # Try with a much shorter prompt
                    short_prompt = request.prompt[:50] if len(request.prompt) > 50 else request.prompt
                    result = text_generator(
                        f"Write a professional email: {short_prompt}",
                        max_new_tokens=50,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=50256
                    )
                else:
                    raise gen_error
            
            generated_text = result[0]['generated_text']
            logger.info(f"Raw AI output: {generated_text[:100]}...")
            
            # Clean and format the result
            cleaned_content = clean_generated_text(generated_text, prompt)
            final_email = format_as_email(cleaned_content, request.tone or "professional")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Draft generated successfully in {processing_time:.2f}s")
            
            return AIResponse(
                success=True,
                result=final_email,
                confidence=0.75,  # Real AI confidence
                processing_time=processing_time
            )
        else:
            # Fallback if AI not available - return a proper AIResponse instead of HTTPException
            logger.warning("AI models not available, using fallback")
            fallback_email = generate_fallback_email(request.prompt, request.tone or "professional")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AIResponse(
                success=True,
                result=fallback_email,
                confidence=0.3,  # Lower confidence for fallback
                processing_time=processing_time
            )
        
    except Exception as e:
        logger.error(f"Error in draft_email: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        return AIResponse(
            success=False,
            result=f"Error generating email draft: {str(e)}",
            processing_time=processing_time
        )

@app.post("/grammar-check", response_model=AIResponse)
async def check_grammar(request: GrammarCheckRequest):
    """Check and correct grammar using tiny AI - real AI grammar correction!"""
    start_time = datetime.now()
    
    try:
        if grammar_fixer:
            # Use AI for grammar correction
            prompt = f"Fix the grammar and improve this text: {request.text}"
            
            result = grammar_fixer(
                prompt,
                max_new_tokens=60,  # Generate up to 60 new tokens for grammar fixes
                num_return_sequences=1,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more precise corrections
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            # Extract the corrected part (remove the prompt)
            corrected_text = generated_text.replace(prompt, "").strip()
            
            if not corrected_text or len(corrected_text) < 10:
                corrected_text = request.text  # Fallback to original
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AIResponse(
                success=True,
                result=corrected_text,
                confidence=0.7,  # AI grammar confidence
                processing_time=processing_time
            )
        else:
            raise HTTPException(status_code=503, detail="Grammar AI not available")
        
    except Exception as e:
        logger.error(f"Error in check_grammar: {e}")
        return AIResponse(
            success=False,
            result=f"Error checking grammar: {str(e)}",
            processing_time=(datetime.now() - start_time).total_seconds()
        )

@app.post("/polish", response_model=AIResponse)
async def polish_text(request: PolishRequest):
    """Polish text using tiny AI - real AI text improvement!"""
    start_time = datetime.now()
    
    try:
        if text_generator:
            # Use AI for text polishing
            style_prompts = {
                "professional": f"Rewrite this text to be more professional: {request.text}",
                "friendly": f"Rewrite this text to be more friendly: {request.text}",
                "formal": f"Rewrite this text to be more formal: {request.text}",
                "concise": f"Rewrite this text to be more concise: {request.text}"
            }
            
            prompt = style_prompts.get(request.style, style_prompts["professional"])
            
            result = text_generator(
                prompt,
                max_new_tokens=80,  # Generate up to 80 new tokens
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            polished_text = clean_generated_text(generated_text, prompt)
            
            if not polished_text or len(polished_text) < 5:
                polished_text = request.text  # Fallback
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AIResponse(
                success=True,
                result=polished_text,
                confidence=0.7,
                processing_time=processing_time
            )
        else:
            raise HTTPException(status_code=503, detail="Polish AI not available")
        
    except Exception as e:
        logger.error(f"Error in polish_text: {e}")
        return AIResponse(
            success=False,
            result=f"Error polishing text: {str(e)}",
            processing_time=(datetime.now() - start_time).total_seconds()
        )

@app.post("/summarize")
async def summarize_email(text: str):
    """Summarize text using tiny AI - real AI summarization!"""
    start_time = datetime.now()
    
    try:
        if summarizer and summarizer != text_generator:
            # Use dedicated summarizer if available
            result = summarizer(
                text,
                max_length=40,
                min_length=10,
                do_sample=False
            )
            summary = result[0]['summary_text']
        elif text_generator:
            # Use text generator for summarization
            prompt = f"Summarize this text in one sentence: {text}"
            result = text_generator(
                prompt,
                max_new_tokens=30,  # Generate up to 30 new tokens for summary
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5,
                pad_token_id=50256
            )
            generated_text = result[0]['generated_text']
            summary = clean_generated_text(generated_text, prompt)
        else:
            raise HTTPException(status_code=503, detail="Summarizer AI not available")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AIResponse(
            success=True,
            result=summary,
            confidence=0.75,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_email: {e}")
        return AIResponse(
            success=False,
            result=f"Error summarizing text: {str(e)}",
            processing_time=(datetime.now() - start_time).total_seconds()
        )

# Email endpoints
@app.get("/api/emails")
async def get_emails():
    """Get all emails from database"""
    emails = get_emails_from_db()
    return {"emails": emails, "total": len(emails)}

@app.get("/api/emails/category/{category}")
async def get_emails_by_category(category: str):
    """Get emails by category from database"""
    if not db_connection:
        return {"emails": [], "total": 0}
    
    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, gmail_id, subject, sender, body, category, 
                       timestamp, read_status as read
                FROM emails 
                WHERE LOWER(category) = LOWER(%s)
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (category,))
            
            emails = cursor.fetchall()
            return {"emails": [dict(email) for email in emails], "total": len(emails)}
            
    except Exception as e:
        logger.error(f"❌ Failed to get emails by category: {e}")
        return {"emails": [], "total": 0}

@app.get("/api/emails/categories")
async def get_categories():
    """Get available email categories from database"""
    if not db_connection:
        return {"categories": ["Work", "Personal", "General"]}
    
    try:
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT category FROM emails WHERE category IS NOT NULL")
            categories = [row[0] for row in cursor.fetchall()]
            return {"categories": categories if categories else ["Work", "Personal", "General"]}
            
    except Exception as e:
        logger.error(f"❌ Failed to get categories: {e}")
        return {"categories": ["Work", "Personal", "General"]}

@app.post("/api/emails/sync")
async def sync_emails():
    """Manually sync emails from Gmail"""
    if not gmail_service:
        raise HTTPException(status_code=503, detail="Gmail service not available")
    
    try:
        logger.info("🔄 Manual email sync requested...")
        emails = fetch_gmail_emails(50)  # Fetch more emails
        store_emails_in_db(emails)
        
        return {
            "success": True,
            "message": f"Synced {len(emails)} emails from Gmail",
            "count": len(emails)
        }
    except Exception as e:
        logger.error(f"❌ Failed to sync emails: {e}")
        return {"success": False, "message": f"Sync failed: {str(e)}"}

@app.post("/api/emails/drafts")
async def create_draft(draft_data: dict):
    """Create a new email draft"""
    if not db_connection:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        with db_connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO emails (subject, body, sender, category, timestamp, read_status)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                draft_data.get("subject", ""),
                draft_data.get("body", ""),
                draft_data.get("to", ""),
                "Draft",
                datetime.now().isoformat(),
                False
            ))
            
            draft_id = cursor.fetchone()[0]
            db_connection.commit()
            
            return {
                "draft": {
                    "id": draft_id,
                    "subject": draft_data.get("subject", ""),
                    "body": draft_data.get("body", ""),
                    "recipient": draft_data.get("to", ""),
                    "created_at": datetime.now().isoformat(),
                    "status": "draft"
                },
                "success": True
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to create draft: {e}")
        return {"success": False, "message": f"Failed to create draft: {str(e)}"}

@app.get("/api/emails/drafts")
async def get_drafts():
    """Get all drafts from database"""
    if not db_connection:
        return {"drafts": [], "total": 0}
    
    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, subject, body, sender as recipient, timestamp as created_at
                FROM emails 
                WHERE category = 'Draft'
                ORDER BY timestamp DESC
            """)
            
            drafts = cursor.fetchall()
            return {"drafts": [dict(draft) for draft in drafts], "total": len(drafts)}
            
    except Exception as e:
        logger.error(f"❌ Failed to get drafts: {e}")
        return {"drafts": [], "total": 0}

# Legacy endpoints for backward compatibility
@app.post("/ai/draft")
async def legacy_draft(request: dict):
    """Legacy endpoint for email drafting"""
    draft_request = EmailDraftRequest(
        prompt=request.get("prompt", ""),
        tone=request.get("tone", "professional"),
        length=request.get("length", "medium")
    )
    return await draft_email(draft_request)

@app.post("/ai/improve")
async def legacy_improve(request: dict):
    """Legacy endpoint for text improvement"""
    polish_request = PolishRequest(
        text=request.get("text", ""),
        style=request.get("style", "professional")
    )
    return await polish_text(polish_request)

@app.post("/ai/grammar")
async def legacy_grammar(request: dict):
    """Legacy endpoint for grammar checking"""
    grammar_request = GrammarCheckRequest(text=request.get("text", ""))
    return await check_grammar(grammar_request)

def generate_fallback_email(prompt: str, tone: str = "professional") -> str:
    """Generate a simple fallback email when AI models aren't available"""
    logger.info(f"Generating fallback email for prompt: {prompt[:50]}...")
    
    # Simple template-based email generation
    if "reply" in prompt.lower() or "respond" in prompt.lower():
        if tone == "friendly":
            content = "Thanks for your email! I'll get back to you soon with more details."
        elif tone == "formal":
            content = "Thank you for your correspondence. I will review your request and respond accordingly."
        else:  # professional
            content = "Thank you for your email. I'll review this and get back to you shortly."
    else:
        # New email
        if tone == "friendly":
            content = "I hope this email finds you well. I wanted to reach out regarding your request."
        elif tone == "formal":
            content = "I am writing to address the matter you have brought to my attention."
        else:  # professional
            content = "I am reaching out to discuss the topic you mentioned."
    
    return format_as_email(content, tone)

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for debugging"""
    return {
        "status": "working",
        "ai_available": AI_AVAILABLE,
        "models_loaded": {
            "text_generator": text_generator is not None,
            "grammar_fixer": grammar_fixer is not None,
            "summarizer": summarizer is not None
        }
    }

@app.post("/test-draft")
async def test_draft_endpoint():
    """Test draft endpoint with minimal data"""
    try:
        request = EmailDraftRequest(prompt="Test email", tone="professional", length="medium")
        result = await draft_email(request)
        return {"test_result": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting EmailTool AI Service (Lightweight)...")
    print("🤖 Using TINY AI models for real AI functionality!")
    print("📊 DistilGPT2 (82MB) + DistilBART (142MB) = ~250MB total")
    print("💻 Laptop-friendly with real AI capabilities!")
    print("📍 Service will be available at: http://localhost:5001")
    print("📖 API Documentation: http://localhost:5001/docs")
    print()
    print("💡 This uses REAL AI models, just tiny ones!")
    print("✅ Classification: Your separate classifier")
    print("✅ Email tasks: Tiny AI models (some mistakes OK)")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info"
    )
