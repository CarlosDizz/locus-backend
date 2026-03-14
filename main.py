import os
import json
import re
import base64
import asyncio
import requests
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import prompts

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import google as livekit_google
from livekit.api import AccessToken, VideoGrants
from livekit import rtc

load_dotenv()

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

chat_histories = {}

app = FastAPI(title="Locus API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    action: str
    roomId: str
    deviceId: str
    context: str = ""
    text: str = ""
    lat: float = None
    lng: float = None

class TokenRequest(BaseModel):
    participant_name: str
    room_name: str
    poi_context: str = ""

@app.post("/get_token")
async def get_token(req: TokenRequest):
    token = AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
    token.with_identity(req.participant_name)
    # Permisos explícitos para que todos puedan publicar y suscribirse
    grant = VideoGrants(
        room_join=True,
        room=req.room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True
    )
    token.with_grants(grant)
    token.with_metadata(req.poi_context)
    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}

# El resto de endpoints (home_chat, etc.) se mantienen igual que los tenías...

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # 1. Esperamos a que entre CUALQUIER participante (anfitrión o invitado)
    # Esto evita el error 1008 al no abrir el flujo de audio al vacío.
    participant = await ctx.wait_for_participant()

    # 2. Configuramos el modelo LATEST que acordamos
    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-latest",
            instructions=prompts.VOICE_SYSTEM_PROMPT,
            voice="Puck"
        )
    )

    agent = Agent(instructions=prompts.VOICE_SYSTEM_PROMPT)
    await session.start(agent=agent, room=ctx.room)

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
            if payload.get("action") == "text_chat":
                # Usamos generate_reply para que Gemini responda al texto
                asyncio.create_task(session.generate_reply(instructions=payload['data']))
            elif payload.get("action") == "image_context":
                image_bytes = base64.b64decode(payload["data"])
                async def process_image():
                    try:
                        desc_resp = gemini_client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=[
                                types.Part.from_bytes(data=image_bytes, mime_type=payload.get("mime_type", "image/jpeg")),
                                types.Part.from_text(text=prompts.VOICE_IMAGE_DESCRIBE)
                            ]
                        )
                        await session.generate_reply(instructions=prompts.VOICE_IMAGE_COMMENT.format(descripcion=desc_resp.text))
                    except: pass
                asyncio.create_task(process_image())
        except: pass

    # 3. Saludo inicial al primer valiente que haya entrado
    await session.generate_reply(instructions=prompts.VOICE_WELCOME_BASE)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))