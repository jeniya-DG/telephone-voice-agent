from common.agent_functions import FUNCTION_DEFINITIONS
from common.prompt_templates import DEEPGRAM_PROMPT_TEMPLATE
from datetime import datetime
import os


VOICE = "aura-2-thalia-en"

# Audio settings
USER_AUDIO_SAMPLE_RATE = 16000
USER_AUDIO_SECS_PER_CHUNK = 0.05
USER_AUDIO_SAMPLES_PER_CHUNK = round(USER_AUDIO_SAMPLE_RATE * USER_AUDIO_SECS_PER_CHUNK)

AGENT_AUDIO_SAMPLE_RATE = 16000
AGENT_AUDIO_BYTES_PER_SEC = 2 * AGENT_AUDIO_SAMPLE_RATE

VOICE_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

AUDIO_SETTINGS = {
    "input": {
        "encoding": "linear16",
        "sample_rate": USER_AUDIO_SAMPLE_RATE,
    },
    "output": {
        "encoding": "linear16",
        "sample_rate": AGENT_AUDIO_SAMPLE_RATE,
        "container": "none",
    },
}

LISTEN_SETTINGS = {
    "provider": {
        "type": "deepgram",
        "model": "flux-general-en",
    }
}

THINK_SETTINGS = {
    "provider": {
        "type": "open_ai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
    },
    "prompt": DEEPGRAM_PROMPT_TEMPLATE.format(documentation=""),
    "functions": FUNCTION_DEFINITIONS,
}

SPEAK_SETTINGS = {
    "provider": {
        "type": "deepgram",
        "model": VOICE,
    }
}

AGENT_SETTINGS = {
    "language": "en",
    "listen": LISTEN_SETTINGS,
    "think": THINK_SETTINGS,
    "speak": SPEAK_SETTINGS,
    "greeting": "",
}

SETTINGS = {"type": "Settings", "audio": AUDIO_SETTINGS, "agent": AGENT_SETTINGS}

# Welcome message templates
WELCOME_MESSAGES = {
    "en": "Hello! I'm {voiceName} from {company}. {capabilities} How can I help you today?",
}


class AgentTemplates:
    def __init__(
        self,
        voiceModel="aura-2-thalia-en",
        voiceName="",
        language="en",
    ):
        self.voiceModel = voiceModel
        if voiceName == "":
            self.voiceName = self.get_voice_name_from_model(self.voiceModel)
        else:
            self.voiceName = voiceName
        self.language = language
        self.company = "Deepgram"

        self.voice_agent_url = VOICE_AGENT_URL
        self.settings = SETTINGS.copy()
        self.settings["agent"] = AGENT_SETTINGS.copy()
        self.settings["agent"]["think"] = THINK_SETTINGS.copy()
        self.settings["agent"]["speak"] = SPEAK_SETTINGS.copy()
        
        self.user_audio_sample_rate = USER_AUDIO_SAMPLE_RATE
        self.user_audio_secs_per_chunk = USER_AUDIO_SECS_PER_CHUNK
        self.user_audio_samples_per_chunk = USER_AUDIO_SAMPLES_PER_CHUNK
        self.agent_audio_sample_rate = AGENT_AUDIO_SAMPLE_RATE
        self.agent_audio_bytes_per_sec = AGENT_AUDIO_BYTES_PER_SEC

        self.personality = f"You are {self.voiceName}, a friendly and professional customer service representative for Deepgram, a Voice API company who provides STT and TTS capabilities via API. Your role is to assist potential customers with general inquiries about Deepgram."
        self.capabilities = "I can help you answer questions about Deepgram."
        self.prompt = DEEPGRAM_PROMPT_TEMPLATE.format(documentation="")

        # Build welcome message
        lang_base = (self.language or "en").split("-")[0].lower()
        welcome_template = WELCOME_MESSAGES.get(lang_base, WELCOME_MESSAGES["en"])
        self.first_message = welcome_template.format(
            voiceName=self.voiceName,
            company=self.company,
            capabilities=self.capabilities,
        )

        # Update settings
        self.settings["agent"]["speak"]["provider"]["model"] = self.voiceModel
        self.settings["agent"]["language"] = self.language
        self.settings["agent"]["think"]["prompt"] = self.personality + "\n\n" + self.prompt
        self.settings["agent"]["greeting"] = self.first_message

    def get_voice_name_from_model(self, model):
        return (
            model.replace("aura-2-", "").replace("aura-", "").split("-")[0].capitalize()
        )
