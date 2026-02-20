"""
Twilio Phone Integration for Deepgram Voice Agent.
Bridges incoming Twilio phone calls to Deepgram Voice Agent API,
reusing the same knowledge base (Kapa), voice switching, and end call functions.

Usage:
    python twilio_server.py

Runs on port 8001 by default. Configure Twilio webhook to POST to /twilio/voice.
"""

import asyncio
import base64
import json
import logging
import os
import random
import time

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response

from common.agent_functions import FUNCTION_MAP
from common.agent_templates import AgentTemplates
from common.config import FILLER_MESSAGES, TWILIO_PORT

load_dotenv(override=True)

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/twilio_server.log"),
    ],
)
logger = logging.getLogger("twilio_server")

app = FastAPI(title="Deepgram Voice Agent - Twilio Phone Integration")


@app.get("/twilio/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "twilio-voice-agent"}


@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    """Webhook for incoming Twilio calls. Returns TwiML to start a media stream."""
    host = request.headers.get("host", "voice.deepgram.com")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/twilio/media" />
    </Connect>
</Response>"""
    logger.info(f"Incoming call - streaming to wss://{host}/twilio/media")
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/twilio/media")
async def twilio_media(websocket: WebSocket):
    """Handle Twilio Media Stream and bridge to Deepgram Voice Agent."""
    await websocket.accept()
    logger.info("Twilio media stream connected")

    dg_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_api_key:
        logger.error("DEEPGRAM_API_KEY not set")
        await websocket.close()
        return

    # Configure agent with Twilio-compatible audio (mulaw, 8kHz)
    agent_templates = AgentTemplates()
    settings = {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "mulaw", "sample_rate": 8000},
            "output": {"encoding": "mulaw", "sample_rate": 8000, "container": "none"},
        },
        "agent": agent_templates.settings["agent"],
    }

    stream_sid = None
    call_active = True
    dg_ws = None

    try:
        # Connect to Deepgram Voice Agent
        dg_ws = await websockets.connect(
            "wss://agent.deepgram.com/v1/agent/converse",
            extra_headers={"Authorization": f"Token {dg_api_key}"},
        )
        await dg_ws.send(json.dumps(settings))
        logger.info("Connected to Deepgram Voice Agent")

        async def twilio_to_deepgram():
            """Forward audio from Twilio phone call to Deepgram."""
            nonlocal stream_sid, call_active
            try:
                while call_active:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "media":
                        audio = base64.b64decode(data["media"]["payload"])
                        if dg_ws and dg_ws.open:
                            await dg_ws.send(audio)
                    elif event == "start":
                        stream_sid = data["start"]["streamSid"]
                        logger.info(f"Stream started: {stream_sid}")
                    elif event == "stop":
                        logger.info("Twilio stream stopped (caller hung up)")
                        call_active = False
                        break
            except WebSocketDisconnect:
                logger.info("Twilio disconnected")
                call_active = False
            except Exception as e:
                logger.error(f"Twilio receiver error: {e}")
                call_active = False

        async def deepgram_to_twilio():
            """Forward audio from Deepgram to Twilio, handle function calls."""
            nonlocal call_active
            filler_complete = asyncio.Event()
            filler_complete.set()
            try:
                async for message in dg_ws:
                    if not call_active:
                        break

                    if isinstance(message, bytes):
                        # Forward agent audio to Twilio
                        if stream_sid:
                            payload = base64.b64encode(message).decode("utf-8")
                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": payload},
                            })

                    elif isinstance(message, str):
                        msg = json.loads(message)
                        msg_type = msg.get("type", "")
                        logger.info(f"DG >>> {message}")

                        # Clear Twilio audio buffer on barge-in
                        if msg_type == "UserStartedSpeaking":
                            if stream_sid:
                                await websocket.send_json({
                                    "event": "clear",
                                    "streamSid": stream_sid,
                                })
                                logger.info("Barge-in: cleared Twilio audio buffer")
                            if not filler_complete.is_set():
                                filler_complete.set()

                        elif msg_type == "AgentAudioDone":
                            if not filler_complete.is_set():
                                filler_complete.set()
                                logger.info("Filler message finished")

                        # Log conversation
                        elif msg_type == "ConversationText":
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            logger.info(f"[{role}] {content}")

                        # Handle function calls
                        elif msg_type == "FunctionCallRequest":
                            for func_call in msg.get("functions", []):
                                function_name = func_call.get("name")
                                function_call_id = func_call.get("id")
                                parameters = json.loads(
                                    func_call.get("arguments", "{}")
                                )
                                logger.info(
                                    f"Function call: {function_name}({parameters})"
                                )

                                func = FUNCTION_MAP.get(function_name)
                                if not func:
                                    logger.error(f"Unknown function: {function_name}")
                                    continue

                                start_time = time.time()

                                if function_name == "end_call":
                                    result = await func(dg_ws, parameters)

                                    response = {
                                        "type": "FunctionCallResponse",
                                        "id": function_call_id,
                                        "name": function_name,
                                        "content": json.dumps(
                                            result["function_response"]
                                        ),
                                    }
                                    await dg_ws.send(json.dumps(response))

                                    inject_msg = result["inject_message"]
                                    await dg_ws.send(json.dumps(inject_msg))
                                    logger.info(f"Farewell: {inject_msg['message']}")

                                    await asyncio.sleep(4)
                                    call_active = False
                                    break

                                elif function_name == "switch_voice":
                                    result = await func(dg_ws, parameters, update_listen=False)
                                    logger.info(
                                        f"Voice switched to: "
                                        f"{result.get('new_voice_name')}"
                                    )
                                    elapsed = time.time() - start_time
                                    logger.info(f"Function {function_name} took {elapsed:.3f}s")

                                    response = {
                                        "type": "FunctionCallResponse",
                                        "id": function_call_id,
                                        "name": function_name,
                                        "content": json.dumps(result),
                                    }
                                    await dg_ws.send(json.dumps(response))
                                    logger.info(f"FunctionCallResponse sent for {function_name}")

                                # All other functions: inject filler, execute concurrently
                                else:
                                    filler = random.choice(FILLER_MESSAGES)
                                    filler_complete.clear()
                                    await dg_ws.send(json.dumps({
                                        "type": "InjectAgentMessage",
                                        "message": filler,
                                    }))
                                    logger.info(f"Filler injected: {filler}")

                                    _func = func
                                    _params = parameters
                                    _call_id = function_call_id
                                    _call_name = function_name
                                    _start = start_time

                                    async def execute_after_filler():
                                        try:
                                            result = await _func(_params)
                                            elapsed = time.time() - _start
                                            logger.info(f"Function {_call_name} took {elapsed:.3f}s")

                                            await filler_complete.wait()

                                            response = {
                                                "type": "FunctionCallResponse",
                                                "id": _call_id,
                                                "name": _call_name,
                                                "content": json.dumps(result),
                                            }
                                            await dg_ws.send(json.dumps(response))
                                        except Exception as e:
                                            logger.error(f"Error in background function: {e}")
                                            error_response = {
                                                "type": "FunctionCallResponse",
                                                "id": _call_id,
                                                "name": _call_name,
                                                "content": json.dumps({"error": str(e)}),
                                            }
                                            await dg_ws.send(json.dumps(error_response))

                                    asyncio.create_task(execute_after_filler())

                        elif msg_type == "Welcome":
                            logger.info(
                                f"Deepgram session: {msg.get('session_id')}"
                            )

                        else:
                            logger.info(f"Unhandled DG message type: {msg_type}")

            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
            ):
                logger.info("Deepgram connection closed")
                call_active = False
            except Exception as e:
                logger.error(f"Deepgram receiver error: {e}")
                call_active = False

        # Run both directions concurrently
        await asyncio.gather(
            twilio_to_deepgram(),
            deepgram_to_twilio(),
            return_exceptions=True,
        )

    except Exception as e:
        logger.error(f"Call error: {e}")
    finally:
        if dg_ws and dg_ws.open:
            await dg_ws.close()
        logger.info("Call ended")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TWILIO_PORT", TWILIO_PORT))
    logger.info(f"Starting Twilio server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
