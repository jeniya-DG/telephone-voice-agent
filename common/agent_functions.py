import json
import os
import httpx
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


async def kapa_query(params):
    """Query Kapa AI knowledge base for documentation answers."""
    question = params.get("question", "")
    if not question:
        return {"error": "question is required"}
    
    project_id = os.environ.get("KAPA_PROJECT_ID")
    api_key = os.environ.get("KAPA_API_KEY", "").strip()
    
    if not project_id or not api_key:
        return {"error": "Kapa credentials not configured", "answer": "I'm sorry, I don't have access to the knowledge base right now."}
    
    url = f"https://api.kapa.ai/query/v1/projects/{project_id}/chat/"
    
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }
    
    payload = {
        "query": question,
    }
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Kapa Query Latency: {elapsed_time:.3f}s")
        
        answer = (data.get("answer") or data.get("response") or "").strip()
        return {
            "success": True,
            "answer": answer or "I couldn't find specific information about that in the documentation.",
            "latency_seconds": round(elapsed_time, 3),
        }
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"Kapa Query Failed after {elapsed_time:.3f}s: {str(e)}")
        return {
            "error": str(e),
            "answer": "I'm having trouble accessing the knowledge base right now. Let me try to help based on what I know.",
        }


async def end_call(websocket, params):
    """End the conversation with an appropriate farewell message."""
    farewell_type = params.get("farewell_type", "general")
    
    if farewell_type == "thanks":
        message = "Thank you for calling! Have a great day!"
    elif farewell_type == "help":
        message = "I'm glad I could help! Have a wonderful day!"
    else:
        message = "Goodbye! Have a nice day!"

    return {
        "function_response": {"status": "closing", "message": message},
        "inject_message": {"type": "InjectAgentMessage", "message": message},
        "close_message": {"type": "close"},
    }


# Function definitions for Voice Agent API
FUNCTION_DEFINITIONS = [
    {
        "name": "kapa_query",
        "description": """REQUIRED: Query the Deepgram knowledge base to answer ANY question about Deepgram.
        
        You MUST call this function whenever the user asks about:
        - Deepgram features (STT, TTS, Voice Agent, etc.)
        - API usage or implementation
        - Pricing or plans
        - Supported languages or models
        - Technical questions
        - How something works
        
        DO NOT answer from memory. ALWAYS call this function first to get accurate information.""",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question to search in the knowledge base.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "end_call",
        "description": """End the conversation when the user says goodbye or indicates they're done.""",
        "parameters": {
            "type": "object",
            "properties": {
                "farewell_type": {
                    "type": "string",
                    "description": "Type of farewell",
                    "enum": ["thanks", "general", "help"],
                }
            },
            "required": ["farewell_type"],
        },
    },
]

# Map function names to implementations
FUNCTION_MAP = {
    "kapa_query": kapa_query,
    "end_call": end_call,
}
