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
from livekit import rtc, api
from livekit.api import AccessToken, VideoGrants

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

VOICE_MODEL = os.environ.get("VOICE_MODEL", "gemini-2.5-flash-native-audio-latest")
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "gemini-3-flash-preview")
CONTENT_MODEL = os.environ.get("CONTENT_MODEL", "gemini-3-flash-preview")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET or not LIVEKIT_URL:
    raise RuntimeError("Faltan credenciales/config de LiveKit")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Locus API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chat_histories = {}
poi_context_cache = {}


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


class EndRoomRequest(BaseModel):
    room_name: str
    requester_role: str
    requester_id: str = ""


class LocusAgent(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_json_loose(raw: str) -> dict:
    if not raw:
        return {}

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}


def recent_turns_to_text(turns: list[dict], limit: int = 10) -> str:
    selected = turns[-limit:]
    if not selected:
        return "(sin historial relevante)"

    lines = []
    for turn in selected:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        lines.append(f"{role.upper()}: {text}")

    return "\n".join(lines)


def append_turn(turns: list[dict], role: str, text: str, max_turns: int = 20):
    text = clean_text(text)
    if not text:
        return

    turns.append({"role": role, "text": text})

    if len(turns) > max_turns:
        del turns[:-max_turns]


def extract_current_poi_name(context_text: str) -> str:
    if not context_text:
        return ""

    patterns = [
        r"Viendo:\s*(.*?)\.\s*Detalles",
        r"Viendo:\s*(.*?)(?:\.|$)",
        r"Lugar actual:\s*(.*?)(?:\.|$)",
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


def get_real_pois(query: str, lat: Optional[float], lng: Optional[float]) -> list[dict]:
    if not MAPS_API_KEY:
        return []

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": MAPS_API_KEY,
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


def generate_text(model: str, prompt_text: str) -> str:
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt_text
        )
        return clean_text(response.text or "")
    except Exception:
        return ""


def generate_json(model: str, prompt_text: str) -> dict:
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return parse_json_loose(response.text or "")
    except Exception:
        return {}


def fetch_wikipedia_summary_es(query: str) -> str:
    if not query:
        return ""

    wikipedia.set_lang("es")

    candidates = [query]
    if not query.lower().startswith("pasaje de "):
        candidates.append(f"Pasaje de {query}")

    for candidate in candidates:
        try:
            return wikipedia.summary(candidate, sentences=6, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            for option in e.options[:5]:
                try:
                    return wikipedia.summary(option, sentences=6, auto_suggest=False)
                except Exception:
                    continue
        except wikipedia.exceptions.PageError:
            continue
        except Exception:
            continue

    return ""


def build_verified_poi_context(poi_name: str, answer_goal: str = "") -> str:
    if not poi_name:
        return ""

    cache_key = f"{poi_name}::{answer_goal}".strip()

    if cache_key in poi_context_cache:
        return poi_context_cache[cache_key]

    raw_text = fetch_wikipedia_summary_es(poi_name)
    if not raw_text:
        return ""

    prompt = prompts.DATA_EXTRACTOR_PROMPT.format(
        poi_name=poi_name,
        answer_goal=answer_goal or "",
        raw_text=raw_text
    )

    verified_context = generate_text(CONTENT_MODEL, prompt)
    if verified_context:
        poi_context_cache[cache_key] = verified_context

    return verified_context


def build_user_turn_text(user_text: str = "", image_description: str = "") -> str:
    parts = []

    user_text = clean_text(user_text)
    image_description = clean_text(image_description)

    if user_text:
        parts.append(f"TEXTO DEL USUARIO: {user_text}")

    if image_description:
        parts.append(f"CONTEXTO VISUAL APORTADO EN ESTE TURNO: {image_description}")

    if not parts:
        parts.append("El usuario quiere comentar o entender algo del POI activo, pero no hay más detalle textual.")

    return "\n".join(parts)


def analyze_turn(active_poi: str, base_context: str, verified_context: str, turns: list[dict], user_turn: str) -> dict:
    prompt = prompts.ORCHESTRATOR_ANALYZE_PROMPT.format(
        active_poi=active_poi or "(sin POI activo)",
        base_context=base_context or "(sin contexto base)",
        verified_context=verified_context or "(sin contexto factual todavía)",
        recent_turns=recent_turns_to_text(turns),
        user_turn=user_turn
    )

    data = generate_json(ORCHESTRATOR_MODEL, prompt)

    if not data:
        return {
            "needs_retrieval": False,
            "reason": "fallback",
            "focus_poi": active_poi or "",
            "retrieval_query": "",
            "bridge_phrase": prompts.VOICE_BRIDGE_FALLBACK.strip(),
            "answer_goal": "Responder de forma prudente y centrada en el POI activo."
        }

    data.setdefault("needs_retrieval", False)
    data.setdefault("reason", "fallback")
    data.setdefault("focus_poi", active_poi or "")
    data.setdefault("retrieval_query", "")
    data.setdefault("bridge_phrase", prompts.VOICE_BRIDGE_FALLBACK.strip())
    data.setdefault("answer_goal", "Responder de forma útil, natural y prudente.")
    return data


def answer_user(
    active_poi: str,
    base_context: str,
    verified_context: str,
    turns: list[dict],
    user_turn: str,
    answer_goal: str
) -> str:
    prompt = prompts.UNIFIED_TURN_ANSWER_PROMPT.format(
        active_poi=active_poi or "(sin POI activo)",
        base_context=base_context or "(sin contexto base)",
        verified_context=verified_context or "(sin contexto factual verificado)",
        recent_turns=recent_turns_to_text(turns),
        user_turn=user_turn,
        answer_goal=answer_goal or "Responder al usuario de forma útil."
    )

    return generate_text(CONTENT_MODEL, prompt)


def ensure_room_state(room_id: str) -> dict:
    if room_id not in chat_histories:
        chat_histories[room_id] = {
            "history": [],
            "lat": None,
            "lng": None,
            "base_context": "",
            "active_poi": "",
            "verified_context": "",
        }
    return chat_histories[room_id]


def get_livekit_api() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )


async def publish_agent_message(ctx: JobContext, text: str):
    text = clean_text(text)
    if not text:
        return

    try:
        payload = json.dumps({
            "action": "assistant_message",
            "text": text
        })
        await ctx.room.local_participant.publish_data(
            payload.encode("utf-8"),
            reliable=True
        )
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/home_chat")
async def home_chat(req: ChatRequest):
    room = ensure_room_state(req.roomId)

    if req.lat is not None and req.lng is not None:
        room["lat"] = req.lat
        room["lng"] = req.lng

    if req.context:
        room["base_context"] = req.context
        extracted_poi = extract_current_poi_name(req.context)
        if extracted_poi:
            room["active_poi"] = extracted_poi

    active_poi = room["active_poi"]
    base_context = room["base_context"]
    history = room["history"]

    if req.action == "setup_profile":
        real_pois = get_real_pois("lugares turísticos", room["lat"], room["lng"])
        nearby_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else "lugares cercanos"

        if active_poi and not room["verified_context"]:
            room["verified_context"] = await asyncio.to_thread(
                build_verified_poi_context,
                active_poi,
                "Dar contexto inicial breve y fiable del POI activo."
            )

        prompt = prompts.CHAT_SETUP_PROMPT.format(
            user_context=req.context or "(sin contexto)",
            active_poi=active_poi or "(sin POI activo)",
            nearby_pois=nearby_pois
        )

        reply = await asyncio.to_thread(generate_text, CONTENT_MODEL, prompt)
        append_turn(history, "assistant", reply)

        pois_block = ""
        if real_pois:
            pois_block = f"\n<POIS>{json.dumps(real_pois, ensure_ascii=False)}</POIS>"

        return {"reply": (reply + pois_block).strip()}

    user_turn = build_user_turn_text(user_text=req.text, image_description="")
    append_turn(history, "user", user_turn)

    real_pois = get_real_pois(req.text or "", room["lat"], room["lng"])
    pois_block = ""
    if real_pois:
        pois_block = f"\n<POIS>{json.dumps(real_pois, ensure_ascii=False)}</POIS>"

    analysis = await asyncio.to_thread(
        analyze_turn,
        active_poi,
        base_context,
        room["verified_context"],
        history,
        user_turn
    )

    if analysis.get("needs_retrieval") and active_poi:
        verified_context = await asyncio.to_thread(
            build_verified_poi_context,
            active_poi,
            analysis.get("answer_goal", req.text or "")
        )
        if verified_context:
            room["verified_context"] = verified_context

    reply = await asyncio.to_thread(
        answer_user,
        active_poi,
        base_context,
        room["verified_context"],
        history,
        user_turn,
        analysis.get("answer_goal", "")
    )

    if not reply:
        reply = "No tengo ese dato confirmado ahora mismo, pero puedo contarte lo que sí sé del lugar."

    append_turn(history, "assistant", reply)
    return {"reply": (reply + pois_block).strip()}


@app.post("/get_token")
async def get_token(req: TokenRequest):
    enriched_context = req.poi_context or ""
    active_poi = extract_current_poi_name(req.poi_context)

    if active_poi:
        verified_context = await asyncio.to_thread(
            build_verified_poi_context,
            active_poi,
            "Preparar contexto inicial fiable para la visita."
        )
        if verified_context:
            enriched_context += f"\n\nCONTEXTO FACTUAL VERIFICADO DEL POI ACTIVO:\n{verified_context}"

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


@app.post("/end_room")
async def end_room(req: EndRoomRequest):
    if req.requester_role != "anfitrion":
        return {"ok": False, "error": "Solo el anfitrión puede cerrar la sala"}

    if not req.room_name:
        return {"ok": False, "error": "Falta room_name"}

    try:
        async with get_livekit_api() as lkapi:
            await lkapi.room.delete_room(
                api.proto_room.DeleteRoomRequest(room=req.room_name)
            )
    except Exception as e:
        msg = str(e)
        if "not found" not in msg.lower() and "does not exist" not in msg.lower():
            return {"ok": False, "error": msg}

    if req.room_name in chat_histories:
        del chat_histories[req.room_name]

    return {"ok": True, "room_name": req.room_name}


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    initial_context = participant.metadata or ""
    active_poi = extract_current_poi_name(initial_context)

    initial_verified_context = ""
    if active_poi:
        initial_verified_context = await asyncio.to_thread(
            build_verified_poi_context,
            active_poi,
            "Preparar contexto inicial fiable para la visita."
        )

    dynamic_prompt = prompts.VOICE_SYSTEM_PROMPT
    dynamic_prompt += f"\n\nPOI ACTIVO ACTUAL:\n{active_poi or '(sin POI activo)'}\n"
    dynamic_prompt += f"\nCONTEXTO BASE DE LA VISITA:\n{initial_context or '(sin contexto base)'}\n"

    if initial_verified_context:
        dynamic_prompt += f"\nCONTEXTO FACTUAL VERIFICADO DEL POI ACTIVO:\n{initial_verified_context}\n"

    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=GEMINI_API_KEY,
            model=VOICE_MODEL,
            instructions=dynamic_prompt,
            voice="Puck"
        )
    )

    agent = LocusAgent(instructions=dynamic_prompt)
    await session.start(agent=agent, room=ctx.room)

    speech_lock = asyncio.Lock()
    turn_lock = asyncio.Lock()

    state = {
        "base_context": initial_context,
        "active_poi": active_poi,
        "verified_context": initial_verified_context,
        "history": [],
    }

    async def speak_text(text: str):
        text = clean_text(text)
        if not text:
            return

        async with speech_lock:
            instructions = (
                "Di exactamente el siguiente texto, sin añadir información nueva, "
                "sin meter otros lugares y manteniendo un tono natural de guía turístico:\n\n"
                f"{text}"
            )
            await session.generate_reply(instructions=instructions)

    async def process_user_turn(
        user_text: str = "",
        image_description: str = "",
        source: str = "text",
        speak_response: bool = True,
        publish_response: bool = True
    ):
        async with turn_lock:
            user_turn = build_user_turn_text(
                user_text=user_text,
                image_description=image_description
            )

            append_turn(state["history"], "user", user_turn)

            analysis = await asyncio.to_thread(
                analyze_turn,
                state["active_poi"],
                state["base_context"],
                state["verified_context"],
                state["history"],
                user_turn
            )

            needs_retrieval = bool(analysis.get("needs_retrieval")) and bool(state["active_poi"])

            if needs_retrieval:
                retrieval_task = asyncio.create_task(
                    asyncio.to_thread(
                        build_verified_poi_context,
                        state["active_poi"],
                        analysis.get("answer_goal", user_text or image_description)
                    )
                )

                bridge_phrase = clean_text(
                    analysis.get("bridge_phrase") or prompts.VOICE_BRIDGE_FALLBACK
                )

                if publish_response:
                    await publish_agent_message(ctx, bridge_phrase)
                if speak_response:
                    await speak_text(bridge_phrase)

                verified_context = await retrieval_task
                if verified_context:
                    state["verified_context"] = verified_context

            final_answer = await asyncio.to_thread(
                answer_user,
                state["active_poi"],
                state["base_context"],
                state["verified_context"],
                state["history"],
                user_turn,
                analysis.get("answer_goal", "")
            )

            if not final_answer:
                final_answer = "No tengo ese dato confirmado con suficiente seguridad, pero puedo seguir ayudándote con este lugar."

            append_turn(state["history"], "assistant", final_answer)

            if publish_response:
                await publish_agent_message(ctx, final_answer)
            if speak_response:
                await speak_text(final_answer)

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(p: rtc.RemoteParticipant):
        if not ctx.room.remote_participants:
            asyncio.create_task(ctx.room.disconnect())

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))

            if payload.get("action") == "text_chat":
                async def handle_text_chat():
                    try:
                        await process_user_turn(
                            user_text=payload.get("data", ""),
                            image_description="",
                            source="text",
                            speak_response=True,
                            publish_response=True
                        )
                    except Exception:
                        pass

                asyncio.create_task(handle_text_chat())

            elif payload.get("action") == "guest_audio":
                async def handle_guest_audio():
                    try:
                        audio_bytes = base64.b64decode(payload["data"])
                        mime_type = payload.get("mime_type", "audio/webm")

                        def transcribe():
                            resp = gemini_client.models.generate_content(
                                model=CONTENT_MODEL,
                                contents=[
                                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                                    types.Part.from_text(
                                        text=(
                                            "Transcribe con precisión lo que dice la persona en este audio. "
                                            "Responde únicamente con el texto transcrito. "
                                            "Si no se entiende, responde vacío."
                                        )
                                    )
                                ]
                            )
                            return clean_text(resp.text or "")

                        transcription = await asyncio.to_thread(transcribe)

                        if transcription:
                            chat_msg = json.dumps({
                                "action": "guest_transcription",
                                "text": transcription
                            })
                            await ctx.room.local_participant.publish_data(
                                chat_msg.encode("utf-8"),
                                reliable=True
                            )

                            await process_user_turn(
                                user_text=transcription,
                                image_description="",
                                source="guest_audio",
                                speak_response=True,
                                publish_response=True
                            )
                    except Exception:
                        pass

                asyncio.create_task(handle_guest_audio())

            elif payload.get("action") == "image_context":
                async def handle_image():
                    try:
                        image_bytes = base64.b64decode(payload["data"])
                        mime_type = payload.get("mime_type", "image/jpeg")
                        text_hint = clean_text(payload.get("text", ""))

                        def describe_image():
                            resp = gemini_client.models.generate_content(
                                model=CONTENT_MODEL,
                                contents=[
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                    types.Part.from_text(text=prompts.VOICE_IMAGE_DESCRIBE)
                                ]
                            )
                            return clean_text(resp.text or "")

                        image_description = await asyncio.to_thread(describe_image)

                        await process_user_turn(
                            user_text=text_hint,
                            image_description=image_description,
                            source="image",
                            speak_response=True,
                            publish_response=True
                        )
                    except Exception:
                        pass

                asyncio.create_task(handle_image())

            elif payload.get("action") == "update_poi_context":
                async def handle_update_poi_context():
                    try:
                        new_context = payload.get("data", "")
                        if not new_context:
                            return

                        state["base_context"] = new_context
                        new_poi = extract_current_poi_name(new_context)

                        if new_poi and new_poi != state["active_poi"]:
                            state["active_poi"] = new_poi
                            state["verified_context"] = await asyncio.to_thread(
                                build_verified_poi_context,
                                new_poi,
                                "Actualizar contexto factual del nuevo POI activo."
                            )
                            state["history"] = []
                    except Exception:
                        pass

                asyncio.create_task(handle_update_poi_context())

        except Exception:
            pass

    try:
        welcome_prompt = prompts.VOICE_WELCOME_PROMPT
        welcome_prompt += f"\n\nPOI ACTIVO:\n{state['active_poi'] or '(sin POI activo)'}"
        welcome_prompt += f"\n\nCONTEXTO BASE:\n{state['base_context'] or '(sin contexto base)'}"
        welcome_text = await asyncio.to_thread(generate_text, CONTENT_MODEL, welcome_prompt)

        if welcome_text:
            append_turn(state["history"], "assistant", welcome_text)
            await publish_agent_message(ctx, welcome_text)
            await speak_text(welcome_text)
    except Exception:
        pass


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
