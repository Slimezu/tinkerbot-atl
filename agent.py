import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
USE_AGENT_SESSION = False
try:
    from livekit.agents import AgentSession, Agent
    USE_AGENT_SESSION = True
except ImportError:
    try:
        from livekit.agents.voice import AgentSession, Agent
        USE_AGENT_SESSION = True
    except ImportError:
        try:
            from livekit.agents.voice import VoicePipelineAgent
        except ImportError:
            try:
                from livekit.agents.pipeline import VoicePipelineAgent
            except ImportError:
                from livekit.agents import VoicePipelineAgent
from livekit.plugins import google, silero

load_dotenv()
logger = logging.getLogger("tinkerbot-agent")

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"healthy","agent":"Tinkerbot ATL Mentor"}')

    def log_message(self, format, *args):
        pass # suppress periodic healthcheck log spam

def start_health_server():
    port_str = os.environ.get("PORT")
    if port_str:
        try:
            port = int(port_str)
            server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            logger.info(f"Health check server listening on 0.0.0.0:{port}")
        except Exception as e:
            logger.warning(f"Could not start health check server: {e}")

SYSTEM_INSTRUCTION = """You are Tinkerbot, an AI mentor for Atal Tinkering Lab (ATL). You were created by Mohammad Daniyal Ahmad and Ridith Shetty. You are a hardware assistant and circuit troubleshooter for students (mostly school age, grades 6-12) who work with Arduino, ESP32, Raspberry Pi, sensors, motors, batteries and basic electronics.

Your job is to help students BUILD, UNDERSTAND and FIX things, and to keep them SAFE. You are a mentor, not just an answer machine: explain the "why" in simple words so the student learns."""

def prewarm(proc: JobProcess):
    # Preload VAD model in memory for instant speech response
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    vad_instance = ctx.proc.userdata.get("vad")
    if vad_instance is None:
        try:
            vad_instance = silero.VAD.load()
        except Exception:
            vad_instance = None

    if USE_AGENT_SESSION:
        logger.info("Initializing LiveKit 1.8 AgentSession with Gemini...")
        agent = Agent(
            instructions=SYSTEM_INSTRUCTION,
        )
        session_kwargs = {
            "stt": google.STT(),
            "llm": google.LLM(
                model="gemini-2.5-flash",
                temperature=0.6,
            ),
            "tts": google.TTS(),
        }
        if vad_instance:
            session_kwargs["vad"] = vad_instance

        session = AgentSession(**session_kwargs)
        await session.start(room=ctx.room, agent=agent)
        try:
            await session.say(
                "Hello! I am Tinkerbot, your ATL hardware lab mentor, created by Mohammad Daniyal Ahmad and Ridith Shetty. What circuit or microcontroller are you working on today?",
                allow_interruptions=True,
            )
        except Exception as e:
            logger.info(f"Greeting dispatch note: {e}")
    else:
        logger.info("Initializing legacy VoicePipelineAgent...")
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=SYSTEM_INSTRUCTION,
        )
        participant = await ctx.wait_for_participant()
        logger.info(f"Starting voice session with participant: {participant.identity}")

        agent = VoicePipelineAgent(
            vad=vad_instance,
            stt=google.STT(),
            llm=google.LLM(
                model="gemini-2.5-flash",
                temperature=0.6,
            ),
            tts=google.TTS(),
            chat_ctx=initial_ctx,
        )
        agent.start(ctx.room, participant)
        await agent.say(
            "Hello! I am Tinkerbot, your ATL hardware lab mentor, created by Mohammad Daniyal Ahmad and Ridith Shetty. What circuit or microcontroller are you working on today?",
            allow_interruptions=True,
        )

if __name__ == "__main__":
    start_health_server()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
