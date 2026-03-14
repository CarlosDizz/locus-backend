import os
import json
import re
import base64
import asyncio
from typing import Optional

import requests
import wikipedia

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

chat_histories = {}

app = FastAPI(title="Locus API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    action: str
    roomId: str
    deviceId: str
    context: str = ""
    text: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None


class TokenRequest(BaseModel):
    participant_name: str
    room_name: str
    poi_context: str = ""


class LocusAgent(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)


def get_real_pois(query: str, lat: Optional[float], lng: Optional[float]) -> list[dict]:
    if not MAPS_API_KEY:
        return []

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": MAPS_API_KEY
    }

    if lat is not None and lng is not None:
        params["location"] = f"{lat},{lng}"
        params["radius"] = 2000

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        results = resp.json().get("results", [])[:3]
        pois = []

        for r in results:
            geometry = r.get("geometry", {}).get("location", {})
            if "lat" in geometry and "lng" in geometry:
                pois.append({
                    "name": r.get("name"),
                    "lat": geometry["lat"],
                    "lng": geometry["lng"],
                    "description": r.get("formatted_address", "")
                })

        return pois
    except Exception:
        return []


def extract_current_poi_name(context_text: str) -> str:
    if not context_text:
        return ""

    patterns = [
        r"Viendo:\s*(.*?)\.\s*Detalles",
        r"Viendo:\s*(.*?)(?:\.|$)",
        r"POI:\s*(.*?)(?:\.|$)",
        r"Lugar:\s*(.*?)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, context_text, flags=re.IGNORECASE)
        if match:
            poi_name = (match.group(1) or "").strip()
            if poi_name:
                return poi_name

    return ""


def strip_need_context(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<NEED_CONTEXT>.*?</NEED_CONTEXT>", "", text, flags=re.DOTALL).strip()


def extract_need_context(text: str) -> Optional[dict]:
    if not text:
        return None

    match = re.search(r"<NEED_CONTEXT>(.*?)</NEED_CONTEXT>", text, flags=re.DOTALL)
    if not match:
        return None

    raw_json = match.group(1).strip()
    try:
        data = json.loads(raw_json)
        if isinstance(data, dict):
            return data
    except Exception:
        return None

    return None


def looks_factual_question(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower().strip()

    triggers = [
        "quién",
        "quien",
        "cuándo",
        "cuando",
        "por qué",
        "porque se llama",
        "cómo se llama",
        "como se llama",
        "quién era",
        "quien era",
        "quién lo hizo",
        "quien lo hizo",
        "quién lo construyó",
        "quien lo construyó",
        "quien lo construyo",
        "quién lo promovió",
        "quien lo promovió",
        "quien lo promovio",
        "arquitecto",
        "promotor",
        "fecha",
        "año",
        "ano",
        "estilo",
        "historia",
        "de dónde viene el nombre",
        "de donde viene el nombre",
        "origen del nombre",
    ]

    return any(trigger in lowered for trigger in triggers)


def fetch_wikipedia_summary_es(query: str) -> str:
    if not query:
        return ""

    try:
        wikipedia.set_lang("es")
        return wikipedia.summary(query, sentences=5, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        if e.options:
            try:
                return wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
            except Exception:
                return ""
        return ""
    except wikipedia.exceptions.PageError:
        return ""
    except Exception:
        return ""


def build_verified_poi_context(poi_name: str, user_question: str = "") -> str:
    if not poi_name:
        return ""

    raw_text = fetch_wikipedia_summary_es(poi_name)
    if not raw_text:
        return ""

    prompt = prompts.DATA_EXTRACTOR_PROMPT.format(
        poi_name=poi_name,
        user_question=user_question or "",
        raw_text=raw_text
    )

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return (resp.text or "").strip()
    except Exception:
        return ""


def build_reactive_context(user_text: str, base_context: str) -> str:
    poi_name = extract_current_poi_name(base_context)

    if not poi_name:
        return ""

    if not looks_factual_question(user_text):
        return ""

    verified_context = build_verified_poi_context(poi_name=poi_name, user_question=user_text)

    if not verified_context:
        return ""

    return f"\n\nCONTEXTO FACTUAL VERIFICADO PARA ESTA RESPUESTA:\n{verified_context}\n"


def maybe_resolve_need_context(raw_reply: str, base_context: str) -> str:
    need_ctx = extract_need_context(raw_reply)
    clean_reply = strip_need_context(raw_reply)

    if not need_ctx:
        return clean_reply

    poi_name = (need_ctx.get("poi") or "").strip() or extract_current_poi_name(base_context)
    question = (need_ctx.get("question") or "").strip()

    if not poi_name:
        return clean_reply

    verified_context = build_verified_poi_context(poi_name=poi_name, user_question=question)

    if not verified_context:
        return clean_reply

    retry_prompt = f"""
Has pedido contexto adicional y ahora ya lo tienes.

CONTEXTO FACTUAL VERIFICADO:
{verified_context}

Responde de nuevo al usuario:
- de forma natural,
- breve,
- exacta,
- sin inventar datos,
- y sin incluir la marca <NEED_CONTEXT>.
""".strip()

    try:
        retry = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=retry_prompt
        )
        return strip_need_context(retry.text or clean_reply)
    except Exception:
        return clean_reply


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/home_chat")
async def home_chat(req: ChatRequest):
    if req.roomId not in chat_histories:
        chat_histories[req.roomId] = {
            "history": [],
            "lat": None,
            "lng": None,
            "context": ""
        }

    room_data = chat_histories[req.roomId]

    if req.lat is not None and req.lng is not None:
        room_data["lat"] = req.lat
        room_data["lng"] = req.lng

    if req.context:
        room_data["context"] = req.context

    history = room_data["history"]
    current_lat = room_data["lat"]
    current_lng = room_data["lng"]
    current_context = room_data["context"]

    pois_block = ""

    if req.action == "setup_profile":
        real_pois = get_real_pois("lugares turísticos", current_lat, current_lng)
        nombres_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else "lugares cercanos"

        prompt = prompts.CHAT_SETUP_PROMPT.format(
            context=req.context,
            nombres_pois=nombres_pois
        )

        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )

        if real_pois:
            pois_block = f"\n<POIS>{json.dumps(real_pois, ensure_ascii=False)}</POIS>"

    else:
        real_pois = get_real_pois(req.text, current_lat, current_lng)
        nombres_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else ""

        prompt = prompts.CHAT_TEXT_PROMPT.format(text=req.text)

        if real_pois:
            prompt += prompts.CHAT_POIS_INSTRUCTION.format(nombres_pois=nombres_pois)
            pois_block = f"\n<POIS>{json.dumps(real_pois, ensure_ascii=False)}</POIS>"
        else:
            prompt += prompts.CHAT_FALLBACK_INSTRUCTION

        reactive_context = build_reactive_context(req.text, current_context)
        if reactive_context:
            prompt += reactive_context

        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history
    )

    raw_text = response.text or ""
    bot_reply = maybe_resolve_need_context(raw_text, current_context)

    if pois_block:
        bot_reply += pois_block

    history.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=bot_reply)]
        )
    )

    return {"reply": bot_reply}


@app.post("/get_token")
async def get_token(req: TokenRequest):
    enriched_context = req.poi_context or ""

    poi_name = extract_current_poi_name(req.poi_context)
    if poi_name:
        try:
            verified_context = build_verified_poi_context(poi_name=poi_name, user_question="")
            if verified_context:
                enriched_context += f"\n\nCONTEXTO FACTUAL VERIFICADO INICIAL:\n{verified_context}"
        except Exception:
            pass

    token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(req.participant_name)
    token.with_name(req.participant_name)
    token.with_metadata(enriched_context)

    grant = VideoGrants(
        room_join=True,
        room=req.room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True
    )
    token.with_grants(grant)

    return {
        "token": token.to_jwt(),
        "ws_url": LIVEKIT_URL
    }


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    user_context = participant.metadata or ""

    dynamic_prompt = prompts.VOICE_SYSTEM_PROMPT
    welcome_msg = prompts.VOICE_WELCOME_BASE

    if user_context:
        dynamic_prompt += f"\n\nCONTEXTO ACTUAL DEL USUARIO:\n{user_context}\n"
        welcome_msg = prompts.VOICE_WELCOME_ENRICHED.format(user_context=user_context)

    dynamic_prompt += "\nNo intentes usar herramientas ni llamadas de función en tiempo real."

    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=GEMINI_API_KEY,
            model="gemini-2.5-flash-native-audio-latest",
            instructions=dynamic_prompt,
            voice="Puck"
        )
    )

    agent = LocusAgent(instructions=dynamic_prompt)
    await session.start(agent=agent, room=ctx.room)

    session_state = {
        "base_context": user_context,
        "last_verified_context": ""
    }

    async def build_reply_instructions_for_text(user_text: str) -> str:
        reactive_context = await asyncio.to_thread(
            build_reactive_context,
            user_text,
            session_state["base_context"]
        )

        if reactive_context:
            session_state["last_verified_context"] = reactive_context

        instructions = prompts.VOICE_TEXT_CHAT.format(text=user_text)

        if reactive_context:
            instructions += "\n" + reactive_context

        return instructions

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
                        user_text = payload.get("data", "")
                        instructions = await build_reply_instructions_for_text(user_text=user_text)
                        await session.generate_reply(instructions=instructions)
                    except Exception:
                        pass

                asyncio.create_task(text_reply())

            elif payload.get("action") == "image_context":
                mime_type = payload.get("mime_type", "image/jpeg")
                image_bytes = base64.b64decode(payload["data"])

                async def process_image():
                    try:
                        def fetch_desc():
                            resp = gemini_client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=[
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                    types.Part.from_text(text=prompts.VOICE_IMAGE_DESCRIBE)
                                ]
                            )
                            return resp.text or ""

                        descripcion = await asyncio.to_thread(fetch_desc)

                        instruccion_foto = prompts.VOICE_IMAGE_COMMENT.format(
                            descripcion=descripcion
                        )

                        if session_state["last_verified_context"]:
                            instruccion_foto += "\n" + session_state["last_verified_context"]

                        await session.generate_reply(instructions=instruccion_foto)
                    except Exception:
                        pass

                asyncio.create_task(process_image())

            elif payload.get("action") == "guest_audio":
                audio_bytes = base64.b64decode(payload["data"])
                mime_type = payload.get("mime_type", "audio/webm")

                async def process_guest_audio():
                    try:
                        resp = gemini_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[
                                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                                types.Part.from_text(
                                    text=(
                                        "Transcribe con precisión lo que dice la persona en este audio. "
                                        "Responde únicamente con el texto transcrito sin comillas. "
                                        "Si es solo ruido o no se entiende, no respondas nada."
                                    )
                                )
                            ]
                        )

                        transcripcion = (resp.text or "").strip()

                        if transcripcion:
                            chat_msg = json.dumps({
                                "action": "guest_transcription",
                                "text": transcripcion
                            })
                            await ctx.room.local_participant.publish_data(
                                chat_msg.encode("utf-8"),
                                reliable=True
                            )

                            instructions = await build_reply_instructions_for_text(user_text=transcripcion)
                            await session.generate_reply(instructions=instructions)
                    except Exception:
                        pass

                asyncio.create_task(process_guest_audio())

            elif payload.get("action") == "update_poi_context":
                async def update_poi_context():
                    try:
                        new_context = payload.get("data", "")
                        if new_context:
                            session_state["base_context"] = new_context

                            poi_name = extract_current_poi_name(new_context)
                            if poi_name:
                                verified_context = await asyncio.to_thread(
                                    build_verified_poi_context,
                                    poi_name,
                                    ""
                                )
                                if verified_context:
                                    session_state["base_context"] += (
                                        f"\n\nCONTEXTO FACTUAL VERIFICADO INICIAL:\n{verified_context}"
                                    )
                                    session_state["last_verified_context"] = (
                                        f"\n\nCONTEXTO FACTUAL VERIFICADO PARA ESTA RESPUESTA:\n{verified_context}\n"
                                    )
                    except Exception:
                        pass

                asyncio.create_task(update_poi_context())

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
