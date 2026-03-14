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

# Cliente de Gemini para chat de texto y descripción de imágenes
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Memoria temporal para el chat de la home
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
    params = {
        "query": query,
        "key": api_key
    }
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
    current_lat = room_data["lat"]
    current_lng = room_data["lng"]

    pois_block = ""

    if req.action == "setup_profile":
        real_pois = get_real_pois("lugares turísticos", current_lat, current_lng)
        nombres_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else "lugares cercanos"

        prompt = prompts.CHAT_SETUP_PROMPT.format(context=req.context, nombres_pois=nombres_pois)
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        if real_pois:
            pois_block = f"\n<POIS>\n{json.dumps(real_pois, ensure_ascii=False)}\n</POIS>"
    else:
        real_pois = get_real_pois(req.text, current_lat, current_lng)
        nombres_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else ""

        prompt = prompts.CHAT_TEXT_PROMPT.format(text=req.text)
        if real_pois:
            prompt += prompts.CHAT_POIS_INSTRUCTION.format(nombres_pois=nombres_pois)
            pois_block = f"\n<POIS>\n{json.dumps(real_pois, ensure_ascii=False)}\n</POIS>"
        else:
            prompt += prompts.CHAT_FALLBACK_INSTRUCTION

        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash', # Texto usa flash normal por velocidad
        contents=history
    )

    bot_reply = re.sub(r'<POIS>.*?</POIS>', '', response.text, flags=re.DOTALL)

    if pois_block:
        bot_reply += pois_block

    history.append(types.Content(role="model", parts=[types.Part.from_text(text=bot_reply)]))

    return {"reply": bot_reply}

@app.post("/get_token")
async def get_token(req: TokenRequest):
    token = AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(req.participant_name)
    token.with_metadata(req.poi_context)

    grant = VideoGrants(
        room_join=True,
        room=req.room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True
    )
    token.with_grants(grant)

    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Esperamos al humano para evitar el error 1008 de Policy Violation
    participant = await ctx.wait_for_participant()

    # Usamos el modelo de AUDIO NATIVO LATEST que decidimos
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
                asyncio.create_task(session.generate_reply(instructions=payload['data']))
            elif payload.get("action") == "image_context":
                image_bytes = base64.b64decode(payload["data"])
                async def process_image():
                    try:
                        desc = gemini_client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=[
                                types.Part.from_bytes(data=image_bytes, mime_type=payload.get("mime_type", "image/jpeg")),
                                types.Part.from_text(text=prompts.VOICE_IMAGE_DESCRIBE)
                            ]
                        ).text
                        await session.generate_reply(instructions=prompts.VOICE_IMAGE_COMMENT.format(descripcion=desc))
                    except: pass
                asyncio.create_task(process_image())
        except: pass

    # Bienvenida inicial
    await session.generate_reply(instructions=prompts.VOICE_WELCOME_BASE)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))