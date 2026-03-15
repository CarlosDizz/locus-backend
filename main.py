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

APP_BUILD = "rome-voice-only-v1"

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

chat_histories: dict[str, dict] = {}
poi_context_cache: dict[str, str] = {}


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


def log(message: str):
    print(f"[LOCUS] {message}", flush=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*+", "", text)
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


def recent_turns_to_text(turns: list[dict], limit: int = 12) -> str:
    selected = turns[-limit:]
    if not selected:
        return "(sin historial relevante)"

    lines = []
    for turn in selected:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)


def append_turn(turns: list[dict], role: str, text: str, max_turns: int = 30):
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
        r"Lugar actual:\s*(.*?)(?:\.|$)",
        r"Viendo:\s*(.*?)\.\s*Detalles",
        r"Viendo:\s*(.*?)(?:\.|$)",
        r"POI:\s*(.*?)(?:\.|$)",
        r"Lugar:\s*(.*?)(?:\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, context_text, flags=re.IGNORECASE)
        if match:
            poi_name = clean_text(match.group(1))
            if poi_name:
                return poi_name
    return ""


def get_real_pois(query: str, lat: Optional[float], lng: Optional[float]) -> list[dict]:
    if not MAPS_API_KEY:
        return []

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": MAPS_API_KEY}

    if lat is not None and lng is not None:
        params["location"] = f"{lat},{lng}"
        params["radius"] = 2000

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []

        results = resp.json().get("results", [])[:3]
        pois = []
        for result in results:
            geometry = result.get("geometry", {}).get("location", {})
            if "lat" in geometry and "lng" in geometry:
                pois.append({
                    "name": result.get("name"),
                    "lat": geometry["lat"],
                    "lng": geometry["lng"],
                    "description": result.get("formatted_address", "")
                })
        return pois
    except Exception as exc:
        log(f"get_real_pois error: {exc}")
        return []


def generate_text(model: str, prompt_text: str) -> str:
    try:
        response = gemini_client.models.generate_content(model=model, contents=prompt_text)
        return clean_text(response.text or "")
    except Exception as exc:
        log(f"generate_text error [{model}]: {exc}")
        return ""


def generate_json(model: str, prompt_text: str) -> dict:
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt_text,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return parse_json_loose(response.text or "")
    except Exception as exc:
        log(f"generate_json error [{model}]: {exc}")
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
        except wikipedia.exceptions.DisambiguationError as exc:
            for option in exc.options[:5]:
                try:
                    return wikipedia.summary(option, sentences=6, auto_suggest=False)
                except Exception:
                    continue
        except wikipedia.exceptions.PageError:
            continue
        except Exception:
            continue
    return ""


def build_verified_context_from_text(subject_name: str, raw_text: str, answer_goal: str = "") -> str:
    if not subject_name or not raw_text:
        return ""

    cache_key = f"RAW::{subject_name}::{answer_goal}".strip()
    if cache_key in poi_context_cache:
        return poi_context_cache[cache_key]

    prompt = prompts.DATA_EXTRACTOR_PROMPT.format(
        poi_name=subject_name,
        answer_goal=answer_goal or "",
        raw_text=raw_text
    )
    verified_context = generate_text(CONTENT_MODEL, prompt)
    if verified_context:
        poi_context_cache[cache_key] = verified_context
    return verified_context


def build_verified_poi_context(poi_name: str, answer_goal: str = "") -> str:
    if not poi_name:
        return ""

    cache_key = f"{poi_name}::{answer_goal}".strip()
    if cache_key in poi_context_cache:
        log(f"build_verified_poi_context cache hit: {cache_key}")
        return poi_context_cache[cache_key]

    log(f"build_verified_poi_context START poi={poi_name} goal={answer_goal}")
    raw_text = fetch_wikipedia_summary_es(poi_name)
    if not raw_text:
        log(f"build_verified_poi_context NO_WIKI poi={poi_name}")
        return ""

    verified_context = build_verified_context_from_text(
        subject_name=poi_name,
        raw_text=raw_text,
        answer_goal=answer_goal
    )
    if verified_context:
        poi_context_cache[cache_key] = verified_context
        log(f"build_verified_poi_context END OK poi={poi_name}")
    else:
        log(f"build_verified_poi_context END EMPTY poi={poi_name}")
    return verified_context


def build_verified_context_from_query(query: str, fallback_name: str, answer_goal: str = "") -> str:
    query = clean_text(query)
    fallback_name = clean_text(fallback_name)

    if not query:
        return build_verified_poi_context(fallback_name, answer_goal)

    cache_key = f"QUERY::{query}::{answer_goal}".strip()
    if cache_key in poi_context_cache:
        log(f"build_verified_context_from_query cache hit: {cache_key}")
        return poi_context_cache[cache_key]

    log(f"build_verified_context_from_query START query={query} fallback={fallback_name}")
    raw_text = fetch_wikipedia_summary_es(query)
    if not raw_text and fallback_name:
        raw_text = fetch_wikipedia_summary_es(fallback_name)
    if not raw_text:
        log("build_verified_context_from_query NO_WIKI")
        return ""

    subject_name = query if len(query) <= 120 else fallback_name
    verified_context = build_verified_context_from_text(
        subject_name=subject_name or fallback_name or "contexto",
        raw_text=raw_text,
        answer_goal=answer_goal
    )
    if verified_context:
        poi_context_cache[cache_key] = verified_context
        log("build_verified_context_from_query END OK")
    else:
        log("build_verified_context_from_query END EMPTY")
    return verified_context


def infer_sub_poi(user_text: str) -> str:
    text = clean_text(user_text).lower()
    if not text:
        return ""

    patterns = [
        r"estoy en (.+)",
        r"estamos en (.+)",
        r"veo (.+)",
        r"tengo delante (.+)",
        r"estoy junto a (.+)",
        r"estoy al lado de (.+)",
        r"quiero saber más del? (.+)",
        r"háblame del? (.+)",
        r"hablame del? (.+)",
        r"sobre el? (.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = clean_text(match.group(1))
            value = re.sub(r"^(el|la|los|las|un|una)\s+", "", value, flags=re.IGNORECASE)
            if value:
                return value
    return ""


def build_user_turn_text(user_text: str = "", image_description: str = "") -> str:
    parts: list[str] = []
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


def answer_user(active_poi: str, base_context: str, verified_context: str, turns: list[dict], user_turn: str, answer_goal: str) -> str:
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
        api_secret=LIVEKIT_API_SECRET,
    )


@app.get("/health")
async def health():
    return {"ok": True, "build": APP_BUILD}


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
        nearby_pois = ", ".join([poi["name"] for poi in real_pois]) if real_pois else "lugares cercanos"

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
        user_turn,
    )

    retrieval_target = analysis.get("retrieval_query", "") or active_poi
    if analysis.get("needs_retrieval") and retrieval_target:
        verified_context = await asyncio.to_thread(
            build_verified_context_from_query,
            retrieval_target,
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

    log(f"BOOT {APP_BUILD}")
    log(f"get_token active_poi={active_poi} context={clean_text(req.poi_context)[:250]}")

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
        can_publish_data=True,
    )
    token.with_grants(grant)

    return {"token": token.to_jwt(), "ws_url": LIVEKIT_URL, "build": APP_BUILD}


@app.post("/end_room")
async def end_room(req: EndRoomRequest):
    if req.requester_role != "anfitrion":
        return {"ok": False, "error": "Solo el anfitrión puede cerrar la sala"}
    if not req.room_name:
        return {"ok": False, "error": "Falta room_name"}

    try:
        async with get_livekit_api() as lkapi:
            await lkapi.room.delete_room(api.proto_room.DeleteRoomRequest(room=req.room_name))
    except Exception as exc:
        msg = str(exc)
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

    log(f"entrypoint build={APP_BUILD} active_poi={active_poi} metadata={clean_text(initial_context)[:300]}")

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
            voice="Puck",
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
        "welcome_token": 0,
        "audio_buffers": {},
        "current_sub_poi": "",
        "seen_user_segments": set(),
        "seen_agent_segments": set(),
    }

    def cleanup_audio_buffers():
        loop_time = asyncio.get_event_loop().time()
        to_delete = []
        for audio_id, item in state["audio_buffers"].items():
            age = loop_time - item["created_at"]
            if age > 120:
                to_delete.append(audio_id)
        for audio_id in to_delete:
            del state["audio_buffers"][audio_id]
            log(f"audio buffer expired audio_id={audio_id}")

    async def speak_text(text: str, token: Optional[int] = None, phase: str = "answer"):
        text = clean_text(text)
        if not text:
            return

        async with speech_lock:
            if token is not None and token != state["welcome_token"]:
                log(f"speak_text SKIP stale token={token} current={state['welcome_token']}")
                return

            log(f"speak_text START phase={phase} text={text[:140]}")
            instructions = (
                "Vas a responder SOLO por voz dentro de una visita guiada compartida. "
                "Di exactamente una respuesta útil y natural al usuario. "
                "No expliques procesos internos. "
                "No menciones que estás buscando, pensando o usando contexto salvo que el texto ya lo diga. "
                "Mantén el foco en el POI activo.\n\n"
                f"TEXTO A DECIR:\n{text}"
            )
            await session.generate_reply(instructions=instructions)
            log(f"speak_text END phase={phase}")

    async def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str) -> str:
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

        return await asyncio.wait_for(asyncio.to_thread(transcribe), timeout=15)

    async def register_participant_turn_only(user_text: str, participant_role: str = "", participant_identity: str = "", segment_id: str = ""):
        text = clean_text(user_text)
        if not text:
            return

        dedupe_key = clean_text(segment_id or f"{participant_identity}:{text[:80]}")
        if dedupe_key and dedupe_key in state["seen_user_segments"]:
            log(f"PARTICIPANT duplicate skipped key={dedupe_key}")
            return

        async with turn_lock:
            if dedupe_key:
                state["seen_user_segments"].add(dedupe_key)

            sub_poi = infer_sub_poi(text)
            if sub_poi:
                state["current_sub_poi"] = sub_poi
                log(f"PARTICIPANT SUB_POI detected={sub_poi}")

            user_turn = build_user_turn_text(user_text=text, image_description="")
            append_turn(state["history"], "user", user_turn)
            log(
                f"PARTICIPANT appended role={participant_role or '(unknown)'} "
                f"identity={participant_identity or '(unknown)'} text={text}"
            )

    async def register_agent_shadow(text: str, segment_id: str = ""):
        text = clean_text(text)
        if not text:
            return
        dedupe_key = clean_text(segment_id or text[:120])
        if dedupe_key and dedupe_key in state["seen_agent_segments"]:
            return
        if dedupe_key:
            state["seen_agent_segments"].add(dedupe_key)
        log(f"AGENT SHADOW ignored_for_ui text={text[:150]}")

    async def process_user_turn(user_text: str = "", image_description: str = "", source: str = "text"):
        async with turn_lock:
            try:
                state["welcome_token"] += 1
                current_token = state["welcome_token"]

                clean_user_text = clean_text(user_text)
                sub_poi = infer_sub_poi(clean_user_text)
                if sub_poi:
                    state["current_sub_poi"] = sub_poi
                    log(f"SUB_POI detected={sub_poi}")
                elif clean_user_text:
                    log(f"SUB_POI keep current={state['current_sub_poi']}")

                user_turn = build_user_turn_text(user_text=user_text, image_description=image_description)
                append_turn(state["history"], "user", user_turn)

                effective_focus = state["active_poi"]
                if state["current_sub_poi"]:
                    effective_focus = f"{state['current_sub_poi']} ({state['active_poi']})" if state["active_poi"] else state["current_sub_poi"]

                log(f"TURN START source={source}")
                log(f"TURN active_poi={state['active_poi']} current_sub_poi={state['current_sub_poi']}")
                log(f"TURN user_turn={user_turn[:300]}")
                log(f"TURN effective_focus={effective_focus}")

                analysis = await asyncio.wait_for(
                    asyncio.to_thread(
                        analyze_turn,
                        effective_focus,
                        state["base_context"],
                        state["verified_context"],
                        state["history"],
                        user_turn,
                    ),
                    timeout=12,
                )
                log(f"analyze_turn END analysis={analysis}")

                retrieval_target = clean_text(analysis.get("retrieval_query", ""))
                if not retrieval_target:
                    retrieval_target = state["current_sub_poi"] or state["active_poi"]

                needs_retrieval = bool(analysis.get("needs_retrieval")) and bool(retrieval_target)
                retrieval_task = None
                if needs_retrieval:
                    log(f"retrieval START target={retrieval_target}")
                    retrieval_task = asyncio.create_task(
                        asyncio.to_thread(
                            build_verified_context_from_query,
                            retrieval_target,
                            state["active_poi"],
                            analysis.get("answer_goal", clean_user_text or image_description),
                        )
                    )

                if retrieval_task:
                    verified_context = await asyncio.wait_for(retrieval_task, timeout=12)
                    if verified_context:
                        state["verified_context"] = verified_context
                        log("retrieval END OK")
                    else:
                        log("retrieval END EMPTY")

                log("answer_user START")
                final_answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        answer_user,
                        effective_focus,
                        state["base_context"],
                        state["verified_context"],
                        state["history"],
                        user_turn,
                        analysis.get("answer_goal", ""),
                    ),
                    timeout=15,
                )
                log(f"answer_user END final_answer={final_answer[:250]}")

                if not final_answer:
                    final_answer = "No tengo ese dato confirmado con suficiente seguridad, pero puedo seguir ayudándote con este lugar."

                append_turn(state["history"], "assistant", final_answer)
                await speak_text(final_answer, token=current_token, phase="answer")
                log("TURN END OK")

            except asyncio.TimeoutError:
                log("TURN TIMEOUT")
                fallback = "Estoy tardando demasiado en afinar esa respuesta. Prueba a preguntármelo de forma más concreta."
                append_turn(state["history"], "assistant", fallback)
                await speak_text(fallback, token=state["welcome_token"], phase="timeout")

            except Exception as exc:
                log(f"TURN ERROR {exc}")
                fallback = "Se me ha cruzado ese turno. Vuelve a decírmelo y te contesto."
                append_turn(state["history"], "assistant", fallback)
                await speak_text(fallback, token=state["welcome_token"], phase="error")

    async def handle_audio_chunk(source: str, payload: dict):
        try:
            cleanup_audio_buffers()

            audio_id = payload.get("audio_id", "")
            chunk_index = int(payload.get("chunk_index", -1))
            total_chunks = int(payload.get("total_chunks", 0))
            mime_type = payload.get("mime_type", "audio/webm")
            chunk_data = payload.get("data", "")

            if not audio_id or chunk_index < 0 or total_chunks <= 0 or not chunk_data:
                log(f"{source}_chunk invalid payload")
                return

            if audio_id not in state["audio_buffers"]:
                state["audio_buffers"][audio_id] = {
                    "source": source,
                    "mime_type": mime_type,
                    "total_chunks": total_chunks,
                    "chunks": {},
                    "created_at": asyncio.get_event_loop().time(),
                }
                log(f"{source}_chunk buffer created audio_id={audio_id} total_chunks={total_chunks}")

            buf = state["audio_buffers"][audio_id]
            buf["chunks"][chunk_index] = chunk_data
            received = len(buf["chunks"])
            log(f"{source}_chunk received audio_id={audio_id} chunk={chunk_index + 1}/{total_chunks}")
            if received < total_chunks:
                return

            ordered = []
            for idx in range(total_chunks):
                part = buf["chunks"].get(idx)
                if part is None:
                    log(f"{source}_chunk missing part audio_id={audio_id} idx={idx}")
                    return
                ordered.append(part)

            base64data = "".join(ordered)
            del state["audio_buffers"][audio_id]

            audio_bytes = base64.b64decode(base64data)
            transcription = await transcribe_audio_bytes(audio_bytes, mime_type)
            log(f"{source}_chunk transcription={transcription}")
            if not transcription:
                return

            if source == "guest_audio":
                chat_msg = json.dumps({"action": "guest_transcription", "text": transcription})
                await ctx.room.local_participant.publish_data(chat_msg.encode("utf-8"), reliable=True)

            await process_user_turn(user_text=transcription, image_description="", source=source)
        except Exception as exc:
            log(f"handle_audio_chunk error source={source}: {exc}")

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(_participant: rtc.RemoteParticipant):
        if not ctx.room.remote_participants:
            asyncio.create_task(ctx.room.disconnect())

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
            action = payload.get("action")
            log(f"data_received action={action}")

            if action == "text_chat":
                async def handle_text_chat():
                    try:
                        text = payload.get("data", "")
                        log(f"TEXT_CHAT recibido text={text}")
                        await process_user_turn(user_text=text, image_description="", source="text")
                    except Exception as exc:
                        log(f"handle_text_chat error: {exc}")
                asyncio.create_task(handle_text_chat())

            elif action == "guest_audio_chunk":
                asyncio.create_task(handle_audio_chunk("guest_audio", payload))

            elif action == "guest_audio":
                asyncio.create_task(handle_audio_chunk("guest_audio", payload))

            elif action in ("participant_transcription", "host_transcription"):
                asyncio.create_task(register_participant_turn_only(
                    payload.get("text", ""),
                    payload.get("role", ""),
                    payload.get("participant_identity", ""),
                    payload.get("segment_id", ""),
                ))

            elif action == "agent_shadow":
                asyncio.create_task(register_agent_shadow(payload.get("text", ""), payload.get("segment_id", "")))

            elif action == "image_context":
                async def handle_image():
                    try:
                        log("image_context START")
                        image_bytes = base64.b64decode(payload["data"])
                        mime_type = payload.get("mime_type", "image/jpeg")
                        text_hint = clean_text(payload.get("text", ""))

                        def describe_image():
                            resp = gemini_client.models.generate_content(
                                model=CONTENT_MODEL,
                                contents=[
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                    types.Part.from_text(text=prompts.VOICE_IMAGE_DESCRIBE),
                                ],
                            )
                            return clean_text(resp.text or "")

                        image_description = await asyncio.wait_for(asyncio.to_thread(describe_image), timeout=15)
                        log(f"image_context description={image_description[:250]}")
                        await process_user_turn(user_text=text_hint, image_description=image_description, source="image")
                    except Exception as exc:
                        log(f"handle_image error: {exc}")
                asyncio.create_task(handle_image())

            elif action == "update_poi_context":
                async def handle_update_poi_context():
                    try:
                        new_context = payload.get("data", "")
                        if not new_context:
                            return

                        log(f"update_poi_context START new_context={new_context[:250]}")
                        state["base_context"] = new_context
                        new_poi = extract_current_poi_name(new_context)
                        if new_poi:
                            state["active_poi"] = new_poi
                            state["current_sub_poi"] = ""
                            state["verified_context"] = await asyncio.wait_for(
                                asyncio.to_thread(
                                    build_verified_poi_context,
                                    new_poi,
                                    "Actualizar contexto factual del nuevo POI activo.",
                                ),
                                timeout=12,
                            )
                            state["history"] = []
                            state["seen_user_segments"].clear()
                            state["seen_agent_segments"].clear()
                            state["welcome_token"] += 1
                            log(f"update_poi_context END active_poi={new_poi}")
                    except Exception as exc:
                        log(f"handle_update_poi_context error: {exc}")
                asyncio.create_task(handle_update_poi_context())

        except Exception as exc:
            log(f"data_received parse error: {exc}")

    try:
        welcome_prompt = prompts.VOICE_WELCOME_PROMPT
        welcome_prompt += f"\n\nPOI ACTIVO:\n{state['active_poi'] or '(sin POI activo)'}"
        welcome_prompt += f"\n\nCONTEXTO BASE:\n{state['base_context'] or '(sin contexto base)'}"
        welcome_text = await asyncio.to_thread(generate_text, CONTENT_MODEL, welcome_prompt)
        if welcome_text:
            append_turn(state["history"], "assistant", welcome_text)
            asyncio.create_task(speak_text(welcome_text, token=state["welcome_token"], phase="welcome"))
    except Exception as exc:
        log(f"welcome error: {exc}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
