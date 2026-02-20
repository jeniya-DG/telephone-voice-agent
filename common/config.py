"""
Shared configuration for Deepgram Voice Agent.
"""

FILLER_MESSAGES = [
    "Let me look that up for you.",
    "One moment while I check.",
    "Sure, let me find that information.",
    "Give me a second to check on that.",
    "Let me pull up that information.",
    "One moment, please.",
    "Sure, let me check on that.",
    "Hang on, let me look into that.",
]

# Default voice settings
DEFAULT_VOICE_MODEL = "aura-2-thalia-en"
DEFAULT_VOICE_NAME = "Thalia"
DEFAULT_LANGUAGE = "en"

# Server ports
WEB_PORT = 8000
TWILIO_PORT = 8001

# CORS origins
CORS_ALLOWED_ORIGINS = [
    "https://voice.deepgram.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
