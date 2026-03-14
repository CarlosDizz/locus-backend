import os
import json
import re
import urllib.parse
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
        model='gemini-2.5-flash',
        contents=history
    )

    bot_reply = re.sub(r'<POIS>.*?</POIS>', '', response.text, flags=re.DOTALL)

    if pois_block:
        bot_reply += pois_block

    history.append(types.Content(role="model", parts=[types.Part.from_text(text=bot_reply)]))

    return {"reply": bot_reply}

class TokenRequest(BaseModel):
    participant_name: str
    room_name: str
    poi_context: str = ""

@app.post("/get_token")
async def get_token(req: TokenRequest):
    enriched_context = req.poi_context
    match = re.search(r'Viendo:\s*(.*?)\.\s*Detalles', req.poi_context)

    if match:
        poi_name = match.group(1)
        try:
            prompt_data = prompts.DATA_EXTRACTOR_PROMPT.format(poi_name=poi_name)
            resp = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_data
            )
            if "NO_DATA" not in resp.text:
                enriched_context += f" | DATOS HISTÓRICOS REALES PARA QUE LOS USES: {resp.text}"
        except:
            pass

    token = AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(req.participant_name)
    token.with_name(req.participant_name)
    token.with_metadata(enriched_context)

    grant = VideoGrants(room_join=True, room=req.room_name)
    token.with_grants(grant)

    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    user_context = participant.metadata or ""

    dynamic_prompt = prompts.VOICE_SYSTEM_PROMPT
    welcome_msg = prompts.VOICE_WELCOME_BASE

    if user_context:
        dynamic_prompt += f"\nCONTEXTO DEL USUARIO: {user_context}."
        welcome_msg = prompts.VOICE_WELCOME_ENRICHED.format(user_context=user_context)

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

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(p: rtc.RemoteParticipant):
        if not ctx.room.remote_participants:
            asyncio.create_task(ctx.room.disconnect())

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
            if payload.get("action") == "text_chat":
                async def text_reply():
                    try:
                        instruccion = prompts.VOICE_TEXT_CHAT.format(text=payload['data'])
                        await session.generate_reply(instructions=instruccion)
                    except Exception:
                        pass
                asyncio.create_task(text_reply())
            elif payload.get("action") == "image_context":
                mime_type = payload.get("mime_type", "image/jpeg")
                image_bytes = base64.b64decode(payload["data"])

                async def process_image():
                    try:
                        def fetch_desc():
                            return gemini_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                    types.Part.from_text(text=prompts.VOICE_IMAGE_DESCRIBE)
                                ]
                            ).text

                        descripcion = await asyncio.to_thread(fetch_desc)
                        instruccion_foto = prompts.VOICE_IMAGE_COMMENT.format(descripcion=descripcion)
                        await session.generate_reply(instructions=instruccion_foto)
                    except Exception:
                        pass

                    asyncio.create_task(process_image())
        except Exception:
            pass

    @ctx.room.on("participant_connected")
    def on_participant_connected(p: rtc.RemoteParticipant):
        welcome_str = prompts.VOICE_NEW_PARTICIPANT_BASE
        if p.metadata:
            welcome_str = prompts.VOICE_NEW_PARTICIPANT_ENRICHED.format(metadata=p.metadata)

        async def send_welcome():
            try:
                await session.generate_reply(instructions=welcome_str)
            except Exception:
                pass

        asyncio.create_task(send_welcome())

    try:
        await session.generate_reply(instructions=welcome_msg)
    except Exception:
        pass

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))