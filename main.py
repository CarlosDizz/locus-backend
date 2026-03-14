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

def get_real_pois(query, lat, lng):
    api_key = os.environ.get("MAPS_API_KEY")
    if not api_key:
        return []

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": api_key}
    if lat and lng:
        params["location"] = f"{lat},{lng}"
        params["radius"] = 2000

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return []

    results = resp.json().get("results", [])[:3]
    pois = []
    for r in results:
        pois.append({
            "name": r.get("name"),
            "lat": r["geometry"]["location"]["lat"],
            "lng": r["geometry"]["location"]["lng"],
            "description": r.get("formatted_address", "")
        })
    return pois

@app.post("/home_chat")
async def home_chat(req: ChatRequest):
    if req.roomId not in chat_histories:
        chat_histories[req.roomId] = {"history": [], "lat": None, "lng": None}

    room_data = chat_histories[req.roomId]
    if req.lat is not None and req.lng is not None:
        room_data["lat"] = req.lat
        room_data["lng"] = req.lng

    history = room_data["history"]

    if req.action == "setup_profile":
        real_pois = get_real_pois("lugares turísticos", room_data["lat"], room_data["lng"])
        prompt = prompts.CHAT_SETUP_PROMPT.format(context=req.context, nombres_pois=", ".join([p["name"] for p in real_pois]))
    else:
        real_pois = get_real_pois(req.text, room_data["lat"], room_data["lng"])
        prompt = prompts.CHAT_TEXT_PROMPT.format(text=req.text)
        if real_pois:
            prompt += prompts.CHAT_POIS_INSTRUCTION.format(nombres_pois=", ".join([p["name"] for p in real_pois]))

    history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=history)
    bot_reply = response.text

    if real_pois:
        bot_reply += f"\n<POIS>\n{json.dumps(real_pois, ensure_ascii=False)}\n</POIS>"

    history.append(types.Content(role="model", parts=[types.Part.from_text(text=bot_reply)]))
    return {"reply": bot_reply}

@app.post("/get_token")
async def get_token(req: TokenRequest):
    token = AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
    token.with_identity(req.participant_name)
    token.with_metadata(req.poi_context) # Inyectamos perfil + info POI

    grant = VideoGrants(room_join=True, room=req.room_name, can_publish=True, can_subscribe=True, can_publish_data=True)
    token.with_grants(grant)
    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    user_context = participant.metadata or ""

    # Sistema dinámico: El guía se centra en el POI y el perfil del usuario
    dynamic_prompt = prompts.VOICE_SYSTEM_PROMPT + f"\nCONTEXTO ACTUAL: {user_context}"

    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-latest",
            instructions=dynamic_prompt,
            voice="Puck"
        )
    )

    agent = Agent(instructions=dynamic_prompt)
    await session.start(agent=agent, room=ctx.room)

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
            if payload.get("action") == "guest_audio":
                audio_bytes = base64.b64decode(payload["data"])
                async def process_guest():
                    resp = gemini_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/webm"),
                                  types.Part.from_text(text="Transcribe el audio. Solo texto.")]
                    )
                    if resp.text:
                        # Enviamos globo al front y respuesta por voz
                        msg = json.dumps({"action": "guest_transcription", "text": resp.text.strip()})
                        await ctx.room.local_participant.publish_data(msg.encode("utf-8"), reliable=True)
                        await session.generate_reply(instructions=prompts.VOICE_TEXT_CHAT.format(text=resp.text))
                asyncio.create_task(process_guest())
        except: pass

    await session.generate_reply(instructions=prompts.VOICE_WELCOME_BASE)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))