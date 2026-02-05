#!/usr/bin/env python3
"""Test script for kapa_query function"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the function
from common.agent_functions import kapa_query


async def test_kapa_query():
    """Test kapa_query with sample questions"""
    
    test_questions = [
        "What is Deepgram?",
        "What TTS voices are available?",
        "How does speech-to-text work?",
        "What languages do you support?",
    ]
    
    print("=" * 60)
    print("Testing kapa_query function")
    print("=" * 60)
    
    # Check if credentials are set
    project_id = os.environ.get("KAPA_PROJECT_ID")
    api_key = os.environ.get("KAPA_API_KEY")
    
    if not project_id or not api_key:
        print("\nERROR: KAPA_PROJECT_ID or KAPA_API_KEY not set in .env")
        return
    
    print(f"\nKapa Project ID: {project_id[:8]}...")
    print(f"Kapa API Key: {api_key[:8]}...")
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("-" * 60)
        
        params = {"question": question}
        result = await kapa_query(params)
        
        if result.get("success"):
            print(f"Latency: {result.get('latency_seconds', 'N/A')}s")
            print(f"Answer: {result.get('answer', 'No answer')[:500]}...")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print()


if __name__ == "__main__":
    asyncio.run(test_kapa_query())
