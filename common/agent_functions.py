import json
import os
import re
import httpx
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


def clean_for_speech(text):
    """Clean text for natural speech - remove parenthetical abbreviations and jargon."""
    # Remove parenthetical content like (STT), (TTS), (API), etc.
    text = re.sub(r'\s*\([A-Z]{2,}\)', '', text)
    # Remove other common parenthetical abbreviations
    text = re.sub(r'\s*\(e\.g\.[^)]*\)', '', text)
    text = re.sub(r'\s*\(i\.e\.[^)]*\)', '', text)
    # Clean up any double spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


async def kapa_query(params):
    """Query Kapa AI knowledge base using retrieval endpoint for document chunks."""
    question = params.get("question", "")
    if not question:
        return {"error": "question is required"}
    
    project_id = os.environ.get("KAPA_PROJECT_ID")
    api_key = os.environ.get("KAPA_API_KEY", "").strip()
    
    if not project_id or not api_key:
        return {"error": "Kapa credentials not configured", "context": ""}
    
    url = f"https://api.kapa.ai/query/v1/projects/{project_id}/retrieval/"
    
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }
    
    payload = {
        "query": question,
        "top_k": 1,  # Get top 1 most relevant chunk (fastest)
    }
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Kapa Retrieval Latency: {elapsed_time:.3f}s")
        
        # Response is a direct array of {source_url, content}
        chunks = data if isinstance(data, list) else []
        
        # Format chunks as context for the LLM (cleaned for natural speech)
        context_parts = []
        for i, chunk in enumerate(chunks[:1], 1):
            content = chunk.get("content", "")
            source_url = chunk.get("source_url", "Documentation")
            if content:
                # Clean content for natural speech
                content = clean_for_speech(content)
                context_parts.append(f"[Source {i}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Retrieved {len(chunks)} chunks from Kapa")
        
        return {
            "success": True,
            "context": context or "No relevant documentation found.",
            "num_chunks": len(chunks),
            "latency_seconds": round(elapsed_time, 3),
        }
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"Kapa Retrieval Failed after {elapsed_time:.3f}s: {str(e)}")
        return {
            "error": str(e),
            "context": "",
        }


# Track current voice state
_current_voice_state = {
    "gender": "female",  # female or male
    "accent": "american",  # american, british, australian
    "model": "aura-2-thalia-en",
}

# Voice options by gender and accent
VOICE_OPTIONS = {
    # English voices
    ("female", "american"): "aura-2-thalia-en",
    ("female", "british"): "aura-2-pandora-en",
    ("female", "australian"): "aura-2-theia-en",
    ("male", "american"): "aura-2-orion-en",
    ("male", "british"): "aura-2-draco-en",
    ("male", "australian"): "aura-2-hyperion-en",
    # Spanish voices
    ("female", "spanish"): "aura-2-estrella-es",
    ("male", "spanish"): "aura-2-nestor-es",
    # French voices
    ("female", "french"): "aura-2-agathe-fr",
    ("male", "french"): "aura-2-hector-fr",
    # German voices
    ("female", "german"): "aura-2-viktoria-de",
    ("male", "german"): "aura-2-julius-de",
    # Italian voices
    ("female", "italian"): "aura-2-livia-it",
    ("male", "italian"): "aura-2-dionisio-it",
    # Dutch voices
    ("female", "dutch"): "aura-2-rhea-nl",
    ("male", "dutch"): "aura-2-sander-nl",
    # Japanese voices
    ("female", "japanese"): "aura-2-izanami-ja",
    ("male", "japanese"): "aura-2-fujin-ja",
}

# Map voice names to models for direct requests
VOICE_NAME_MAP = {
    # English - Featured
    "thalia": "aura-2-thalia-en",
    "andromeda": "aura-2-andromeda-en",
    "helena": "aura-2-helena-en",
    "apollo": "aura-2-apollo-en",
    "arcas": "aura-2-arcas-en",
    "aries": "aura-2-aries-en",
    # English - British/Australian
    "draco": "aura-2-draco-en",
    "pandora": "aura-2-pandora-en",
    "hyperion": "aura-2-hyperion-en",
    "theia": "aura-2-theia-en",
    # English - Other popular
    "luna": "aura-2-luna-en",
    "orion": "aura-2-orion-en",
    "athena": "aura-2-athena-en",
    "zeus": "aura-2-zeus-en",
    "aurora": "aura-2-aurora-en",
    "iris": "aura-2-iris-en",
    # Spanish
    "estrella": "aura-2-estrella-es",
    "celeste": "aura-2-celeste-es",
    "nestor": "aura-2-nestor-es",
    # French
    "agathe": "aura-2-agathe-fr",
    "hector": "aura-2-hector-fr",
    # German
    "viktoria": "aura-2-viktoria-de",
    "julius": "aura-2-julius-de",
    # Italian
    "livia": "aura-2-livia-it",
    "dionisio": "aura-2-dionisio-it",
    # Dutch
    "rhea": "aura-2-rhea-nl",
    "sander": "aura-2-sander-nl",
    # Japanese
    "izanami": "aura-2-izanami-ja",
    "fujin": "aura-2-fujin-ja",
}

def get_voice_name(model):
    """Extract friendly name from model string."""
    # Remove prefix and language suffix
    name = model.replace("aura-2-", "")
    name = name.rsplit("-", 1)[0]  # Remove -en, -es, -fr, etc.
    return name.capitalize()


async def switch_voice(websocket, params):
    """Switch the TTS voice based on user request, remembering previous gender/accent."""
    global _current_voice_state
    
    voice_type = params.get("voice_type", "").lower()
    selected_voice = None
    
    # First check if user requested a specific voice name
    for name, model in VOICE_NAME_MAP.items():
        if name in voice_type:
            selected_voice = model
            # Infer gender from model for state tracking
            feminine_voices = ["thalia", "andromeda", "helena", "pandora", "theia", "luna", 
                             "athena", "aurora", "iris", "estrella", "celeste", "agathe",
                             "viktoria", "livia", "rhea", "izanami"]
            _current_voice_state["gender"] = "female" if name in feminine_voices else "male"
            break
    
    if not selected_voice:
        # Determine requested gender and accent
        requested_gender = _current_voice_state["gender"]
        requested_accent = _current_voice_state["accent"]
        
        # Check for gender request
        if "male" in voice_type and "female" not in voice_type:
            requested_gender = "male"
        elif "female" in voice_type or "woman" in voice_type:
            requested_gender = "female"
        
        # Check for accent/language request
        if "british" in voice_type or "uk" in voice_type or "english" in voice_type:
            requested_accent = "british"
        elif "australian" in voice_type or "aussie" in voice_type:
            requested_accent = "australian"
        elif "american" in voice_type or "us" in voice_type:
            requested_accent = "american"
        elif "spanish" in voice_type or "español" in voice_type:
            requested_accent = "spanish"
        elif "french" in voice_type or "français" in voice_type:
            requested_accent = "french"
        elif "german" in voice_type or "deutsch" in voice_type:
            requested_accent = "german"
        elif "italian" in voice_type or "italiano" in voice_type:
            requested_accent = "italian"
        elif "dutch" in voice_type or "netherlands" in voice_type:
            requested_accent = "dutch"
        elif "japanese" in voice_type or "日本語" in voice_type:
            requested_accent = "japanese"
        
        # Get the voice model for this combination
        selected_voice = VOICE_OPTIONS.get(
            (requested_gender, requested_accent),
            "aura-2-thalia-en"
        )
        
        # Update state
        _current_voice_state["gender"] = requested_gender
        _current_voice_state["accent"] = requested_accent
    
    _current_voice_state["model"] = selected_voice
    new_voice_name = get_voice_name(selected_voice)
    
    # Switch the voice immediately
    update_message = {
        "type": "UpdateSpeak",
        "speak": {
            "provider": {
                "type": "deepgram",
                "model": selected_voice
            }
        }
    }
    await websocket.send(json.dumps(update_message))
    logger.info(f"Voice switched to: {selected_voice}")
    
    # Return info for LLM to respond naturally
    return {
        "success": True,
        "new_voice_name": new_voice_name,
        "message": f"Voice switched. You are now {new_voice_name}. Introduce yourself briefly.",
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
        "description": """REQUIRED: Retrieve relevant documentation from the Deepgram knowledge base.
        
        You MUST call this function whenever the user asks about:
        - Deepgram features (STT, TTS, Voice Agent, etc.)
        - API usage or implementation
        - Pricing or plans
        - Supported languages or models
        - Technical questions
        - How something works
        
        This function returns context from documentation. Use the returned context to formulate your answer.
        DO NOT answer from memory. ALWAYS call this function first to get accurate documentation context.""",
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
        "name": "switch_voice",
        "description": """Call this when user asks to change voice/accent/language, OR when user speaks in a different language.
        
        Supported languages: american, british, australian, spanish, french, german, italian, dutch, japanese
        
        IMPORTANT: If user speaks Spanish, French, German, etc., call this function FIRST, then respond in their language.
        
        After calling, respond in the new language (if applicable) and introduce yourself.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "voice_type": {
                    "type": "string",
                    "description": "Voice/language request (e.g., 'spanish', 'french', 'british male')",
                }
            },
            "required": ["voice_type"],
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
    "switch_voice": switch_voice,
    "end_call": end_call,
}
