"""
Test script for Lenny's Wisdom MCP Server.
Tests the core functions without the MCP protocol.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
import google.generativeai as genai

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")
GEMINI_API_KEY = os.getenv("GEMIMI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

print("=" * 50)
print("Testing Lenny's Wisdom MCP Server")
print("=" * 50)

# Test 1: Supabase connection
print("\n1. Testing Supabase connection...")
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = supabase.table("episodes").select("title, slug").limit(3).execute()
    print(f"   ✓ Connected! Found {len(result.data)} episodes:")
    for ep in result.data:
        print(f"     - {ep['title'][:50]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Gemini embeddings
print("\n2. Testing Gemini embeddings...")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    test_text = "How should I structure my product team?"
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=test_text,
        task_type="retrieval_query"
    )
    embedding = result["embedding"]
    print(f"   ✓ Generated embedding with {len(embedding)} dimensions")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Semantic search
print("\n3. Testing semantic search...")
try:
    # Get embedding for query
    query = "product management best practices"
    query_result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = query_result["embedding"]
    
    # Search using the function we created
    search_result = supabase.rpc(
        "search_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": 3
        }
    ).execute()
    
    if search_result.data:
        print(f"   ✓ Found {len(search_result.data)} matching chunks:")
        for chunk in search_result.data:
            print(f"     - [{chunk['guest_name']}] {chunk['content'][:80]}...")
            print(f"       Similarity: {chunk['similarity']:.3f}")
    else:
        print("   ⚠ No chunks found. Did the ingestion complete?")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Check data counts
print("\n4. Checking data counts...")
try:
    guests = supabase.table("guests").select("id", count="exact").execute()
    episodes = supabase.table("episodes").select("id", count="exact").execute()
    chunks = supabase.table("transcript_chunks").select("id", count="exact").execute()
    
    print(f"   Guests: {guests.count}")
    print(f"   Episodes: {episodes.count}")
    print(f"   Chunks: {chunks.count}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 50)
print("Tests complete!")
print("=" * 50)
