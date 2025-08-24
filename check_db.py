#!/usr/bin/env python3
"""
Quick script to check the database structure and content
"""

import psycopg2
from psycopg2.extras import RealDictCursor

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'database': 'emailTool',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

def check_database():
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("=== DATABASE CONNECTION SUCCESSFUL ===")
        print()
        
        # Check what tables exist
        print("=== TABLES IN DATABASE ===")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        for table in tables:
            print(f"- {table['table_name']}")
        print()
        
        # Check emails table structure
        print("=== EMAILS TABLE STRUCTURE ===")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'emails' 
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        if columns:
            print("Columns:")
            for col in columns:
                print(f"- {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']}, default: {col['column_default']})")
        else:
            print("No emails table found or no columns!")
        print()
        
        # Check how many emails we have
        print("=== EMAILS TABLE DATA ===")
        try:
            cursor.execute("SELECT COUNT(*) as count FROM emails")
            count = cursor.fetchone()
            print(f"Total emails: {count['count']}")
            
            # Show first few emails
            cursor.execute("SELECT * FROM emails LIMIT 3")
            emails = cursor.fetchall()
            if emails:
                print("\nFirst 3 emails:")
                for i, email in enumerate(emails, 1):
                    print(f"{i}. ID: {email.get('id', 'N/A')}")
                    print(f"   Subject: {email.get('subject', 'N/A')}")
                    print(f"   Sender: {email.get('sender', 'N/A')}")
                    print(f"   Available fields: {list(email.keys())}")
                    print()
            else:
                print("No emails found in table")
                
        except Exception as e:
            print(f"Error reading emails: {e}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__":
    check_database()
