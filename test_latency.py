"""
Latency Test Script for Deepgram Voice Agent Pipeline.

Connects to the Voice Agent via microphone and measures timing at each stage:
  - STT: UserStartedSpeaking → ConversationText (user)
  - LLM Think: ConversationText (user) → FunctionCallRequest or ConversationText (assistant)
  - Function (Kapa): FunctionCallRequest → FunctionCallResponse sent
  - TTS First Byte: ConversationText (assistant) → first audio byte
  - Total Turn: ConversationText (user) → first audio byte

Usage:
    python test_latency.py
    (Speak into your microphone, latency is printed after each turn)
"""

import asyncio
import json
import os
import time
import pyaudio
import websockets
from dotenv import load_dotenv
from common.agent_functions import FUNCTION_MAP
from common.agent_templates import AgentTemplates, AGENT_AUDIO_SAMPLE_RATE

load_dotenv()

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ms(seconds):
    """Convert seconds to milliseconds string."""
    return f"{seconds * 1000:.0f}ms"


class LatencyTracker:
    def __init__(self):
        self.reset()
        self.turn_count = 0

    def reset(self):
        """Reset all timestamps for a new turn."""
        self.user_started_speaking = None
        self.user_text_time = None
        self.user_text = ""
        self.function_call_time = None
        self.function_name = None
        self.function_response_time = None
        self.function_latency = None  # From agent_functions (e.g., Kapa API time)
        self.assistant_first_text_time = None
        self.assistant_text = ""
        self.first_audio_byte_time = None
        self.agent_audio_done_time = None

    def print_latency(self):
        """Print latency breakdown for this turn."""
        self.turn_count += 1
        print(f"\n{BOLD}{'='*60}")
        print(f"  TURN {self.turn_count} LATENCY BREAKDOWN")
        print(f"{'='*60}{RESET}")

        if self.user_text:
            print(f"  {CYAN}User:{RESET} \"{self.user_text}\"")
        if self.assistant_text:
            print(f"  {GREEN}Agent:{RESET} \"{self.assistant_text[:100]}{'...' if len(self.assistant_text) > 100 else ''}\"")

        print(f"\n  {BOLD}Stage                    Latency{RESET}")
        print(f"  {'-'*45}")

        # STT: UserStartedSpeaking → ConversationText (user)
        if self.user_started_speaking and self.user_text_time:
            stt = self.user_text_time - self.user_started_speaking
            print(f"  STT (speech→text)      {YELLOW}{ms(stt):>8}{RESET}")

        # LLM Think: user text → function call or assistant text
        if self.user_text_time:
            if self.function_call_time:
                think = self.function_call_time - self.user_text_time
                print(f"  LLM Think (→func call) {YELLOW}{ms(think):>8}{RESET}")
            elif self.assistant_first_text_time:
                think = self.assistant_first_text_time - self.user_text_time
                print(f"  LLM Think (→response)  {YELLOW}{ms(think):>8}{RESET}")

        # Function execution time
        if self.function_call_time and self.function_response_time:
            func_time = self.function_response_time - self.function_call_time
            label = f"Function ({self.function_name})"
            print(f"  {label:<24} {YELLOW}{ms(func_time):>8}{RESET}")
            if self.function_latency:
                print(f"    └─ API call            {CYAN}{ms(self.function_latency):>8}{RESET}")

        # LLM after function: function response → assistant text
        if self.function_response_time and self.assistant_first_text_time:
            post_func = self.assistant_first_text_time - self.function_response_time
            print(f"  LLM (func→response)    {YELLOW}{ms(post_func):>8}{RESET}")

        # TTS: assistant text → first audio byte
        if self.assistant_first_text_time and self.first_audio_byte_time:
            tts = self.first_audio_byte_time - self.assistant_first_text_time
            print(f"  TTS (text→first audio) {YELLOW}{ms(tts):>8}{RESET}")

        # Audio duration
        if self.first_audio_byte_time and self.agent_audio_done_time:
            audio_dur = self.agent_audio_done_time - self.first_audio_byte_time
            print(f"  Audio playback         {CYAN}{ms(audio_dur):>8}{RESET}")

        # Total turnaround: user text → first audio byte
        print(f"  {'-'*45}")
        if self.user_text_time and self.first_audio_byte_time:
            total = self.first_audio_byte_time - self.user_text_time
            color = GREEN if total < 2 else YELLOW if total < 4 else RED
            print(f"  {BOLD}TOTAL (user text→audio)  {color}{ms(total):>8}{RESET}")

        # End-to-end: user started speaking → first audio
        if self.user_started_speaking and self.first_audio_byte_time:
            e2e = self.first_audio_byte_time - self.user_started_speaking
            print(f"  {BOLD}E2E (speak→audio)        {CYAN}{ms(e2e):>8}{RESET}")

        print()


async def run_latency_test():
    dg_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_api_key:
        print(f"{RED}Error: DEEPGRAM_API_KEY not set{RESET}")
        return

    agent_templates = AgentTemplates()
    settings = agent_templates.settings

    print(f"\n{BOLD}Deepgram Voice Agent Latency Tester{RESET}")
    print(f"STT Model: {settings['agent']['listen']['provider']['model']}")
    print(f"LLM Model: {settings['agent']['think']['provider']['model']}")
    print(f"TTS Voice: {settings['agent']['speak']['provider']['model']}")
    print(f"\n{CYAN}Speak into your microphone. Latency is measured after each turn.{RESET}")
    print(f"{CYAN}Press Ctrl+C to stop.{RESET}\n")

    # Connect to Deepgram
    ws = await websockets.connect(
        agent_templates.voice_agent_url,
        extra_headers={"Authorization": f"Token {dg_api_key}"},
    )
    await ws.send(json.dumps(settings))

    tracker = LatencyTracker()

    # Audio setup
    audio = pyaudio.PyAudio()
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def audio_callback(input_data, frame_count, time_info, status_flag):
        loop.call_soon_threadsafe(audio_queue.put_nowait, input_data)
        return (input_data, pyaudio.paContinue)

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=agent_templates.user_audio_sample_rate,
        input=True,
        frames_per_buffer=agent_templates.user_audio_samples_per_chunk,
        stream_callback=audio_callback,
    )
    stream.start_stream()

    async def sender():
        while True:
            data = await audio_queue.get()
            try:
                await ws.send(data)
            except Exception:
                break

    async def receiver():
        got_first_audio_this_turn = False

        async for message in ws:
            now = time.time()

            if isinstance(message, bytes):
                if not got_first_audio_this_turn:
                    tracker.first_audio_byte_time = now
                    got_first_audio_this_turn = True
                continue

            msg = json.loads(message)
            msg_type = msg.get("type", "")

            if msg_type == "UserStartedSpeaking":
                # New turn starting - if we had a previous turn, print it
                if tracker.user_text:
                    # Don't reset yet, let AgentAudioDone handle printing
                    pass
                tracker.user_started_speaking = now
                got_first_audio_this_turn = False

            elif msg_type == "ConversationText":
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    tracker.user_text_time = now
                    tracker.user_text = content
                    # Reset turn tracking for new response
                    tracker.function_call_time = None
                    tracker.function_name = None
                    tracker.function_response_time = None
                    tracker.function_latency = None
                    tracker.assistant_first_text_time = None
                    tracker.assistant_text = ""
                    tracker.first_audio_byte_time = None
                    tracker.agent_audio_done_time = None
                    got_first_audio_this_turn = False
                    print(f"  {CYAN}> {content}{RESET}")

                elif role == "assistant":
                    if not tracker.assistant_first_text_time:
                        tracker.assistant_first_text_time = now
                    tracker.assistant_text += content + " "

            elif msg_type == "FunctionCallRequest":
                tracker.function_call_time = now
                for func_call in msg.get("functions", []):
                    function_name = func_call.get("name")
                    function_call_id = func_call.get("id")
                    parameters = json.loads(func_call.get("arguments", "{}"))
                    tracker.function_name = function_name
                    print(f"  {YELLOW}>> {function_name}({json.dumps(parameters)}){RESET}")

                    func = FUNCTION_MAP.get(function_name)
                    if func:
                        if function_name in ("end_call", "switch_voice"):
                            result = await func(ws, parameters)
                        else:
                            result = await func(parameters)

                        # Extract Kapa latency if available
                        if isinstance(result, dict) and "latency_seconds" in result:
                            tracker.function_latency = result["latency_seconds"]

                        tracker.function_response_time = time.time()

                        # Handle end_call specially
                        if function_name == "end_call":
                            response = {
                                "type": "FunctionCallResponse",
                                "id": function_call_id,
                                "name": function_name,
                                "content": json.dumps(result.get("function_response", result)),
                            }
                        else:
                            response = {
                                "type": "FunctionCallResponse",
                                "id": function_call_id,
                                "name": function_name,
                                "content": json.dumps(result),
                            }
                        await ws.send(json.dumps(response))

            elif msg_type == "AgentAudioDone":
                tracker.agent_audio_done_time = now
                if tracker.user_text:
                    tracker.print_latency()

            elif msg_type == "Welcome":
                print(f"  {GREEN}Connected: session {msg.get('session_id')}{RESET}\n")

    try:
        await asyncio.gather(sender(), receiver())
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        await ws.close()
        print(f"\n{BOLD}Done.{RESET}")


if __name__ == "__main__":
    try:
        asyncio.run(run_latency_test())
    except KeyboardInterrupt:
        print(f"\n{BOLD}Stopped.{RESET}")
