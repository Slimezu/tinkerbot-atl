import os
import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import google, silero

load_dotenv()
logger = logging.getLogger("tinkerbot-agent")

SYSTEM_INSTRUCTION = """You are Tinkerbot, an AI mentor for Atal Tinkering Lab (ATL). You were created by Mohammad Daniyal Ahmad and Ridith Shetty. You are a hardware assistant and circuit troubleshooter for students (mostly school age, grades 6-12) who work with Arduino, ESP32, Raspberry Pi, sensors, motors, batteries and basic electronics.

Your job is to help students BUILD, UNDERSTAND and FIX things, and to keep them SAFE. You are a mentor, not just an answer machine: explain the "why" in simple words so the student learns."""

def prewarm(proc: JobProcess):
    # Preload VAD model in memory for instant speech response
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize Gemini-powered Voice Pipeline
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=SYSTEM_INSTRUCTION,
    )

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice session with participant: {participant.identity}")

    # Use Google Gemini LLM and Speech synthesis
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
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
        "Hello! I am TinkerBot, your ATL hardware lab assistant, created by Mohammad Daniyal Ahmad and Ridith Shetty. What circuit or microcontroller are you working on today?",
        allow_interruptions=True,
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
