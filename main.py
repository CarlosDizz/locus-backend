import os
import io
import json
import wave
import tempfile
import urllib.parse
import urllib.request
import logging
import base64
import asyncio
import re
from typing import Optional

from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocusIA")

MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "cedar")
OPENAI_TRANSCRIBE_MODEL = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

FRONT_AUDIO_SAMPLE_RATE = 16000
FRONT_AUDIO_CHANNELS = 1
FRONT_AUDIO_BYTES_PER_SAMPLE = 2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# MAPS
# ---------------------------------------------------------

def buscar_nuevo_lugar(consulta: str, lat: float, lng: float) -> str:
    if not MAPS_API_KEY:
        logger.warning("MAPS_API_KEY no configurada")
        return "[]"

    try:
        query = urllib.parse.quote(consulta)
        url = (
            "https://maps.googleapis.com/maps/api/place/textsearch/json"
            f"?query={query}"
            f"&location={lat},{lng}"
            f"&radius=3000"
            f"&key={MAPS_API_KEY}"
            f"&language=es"
        )
        req = urllib.request.Request(url)

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = data.get("results", [])[:4]

            payload = []
            for r in results:
                geometry = r.get("geometry", {}).get("location", {})
                if "lat" in geometry and "lng" in geometry:
                    payload.append(
                        {
                            "name": r.get("name"),
                            "lat": geometry["lat"],
                            "lng": geometry["lng"],
                        }
                    )

            return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.exception(f"Error en Maps: {e}")
        return "[]"


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------

def build_home_context(user_ctx: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""
Eres Locus, un guía turístico experto, directo y conversacional.

Perfil de los viajeros:
{user_ctx}

Ubicación aproximada:
Latitud {lat}, longitud {lng}

Lugares cercanos detectados:
{pois_data}

Reglas:
- Habla en el idioma del usuario o en el idioma que te pidan.
- Si el usuario pide una variedad regional o un acento, adáptalo de forma natural.
- Responde de forma clara, breve y útil.
- No uses coletillas como "qué interesante" o "buena pregunta".
- No inventes datos concretos.
- Si no estás seguro de algo, dilo claramente.
- No des enlaces ni menciones URLs.
- Si el usuario pregunta qué hay cerca o acaba de llegar, incluye SIEMPRE un bloque <POIS> válido.
- El formato obligatorio para el mapa es exactamente:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
""".strip()


def build_voice_context(
    user_ctx: str,
    lat: float,
    lng: float,
    pois_data: str,
    poi_name: str = "",
    poi_research_summary: str = "",
    last_image_summary: str = "",
) -> str:
    poi_line = f"POI actual de la visita: {poi_name}." if poi_name else "No hay POI actual fijado."
    research_block = poi_research_summary.strip() or "Sin dossier ampliado todavía."
    image_block = last_image_summary.strip() or "Sin contexto visual reciente."

    return f"""
Eres Locus, un guía turístico presencial que acompaña a los viajeros durante una visita real.

Perfil del viajero:
{user_ctx}

Ubicación aproximada:
Latitud {lat}, longitud {lng}

Lugares cercanos:
{pois_data}

{poi_line}

CONTEXTO PRIVADO DEL GUÍA (NO DEBES MENCIONARLO):
{research_block}

ÚLTIMO CONTEXTO VISUAL (NO DEBES MENCIONAR QUE LO TIENES COMO SISTEMA):
{image_block}

Reglas:
- Habla directamente al visitante como un guía real.
- Nunca menciones dossier, notas, research, fuentes, contexto interno, preparación o guion.
- No hables como un asistente que prepara material para otro.
- Tú eres el guía.

- Habla en el idioma del usuario o en el idioma que te pidan.
- Si el usuario pide una variedad regional o un acento, adáptalo de forma natural, sin caricatura.
- Por defecto responde en 2 o 3 frases útiles.
- Si el usuario pide más detalle, puedes ampliar.
- Responde primero a la pregunta concreta del usuario.
- Si existe un POI actual, úsalo siempre como contexto principal.
- Si el usuario menciona algo que tiene delante, intenta conectarlo con el POI actual y con el contexto visual.

- No inventes datos.
- No afirmes fechas, nombres, autores, materiales, estilos o hechos históricos sin base suficiente.
- Si no estás seguro, dilo claramente.
- Si hace falta más contexto, pide foto, placa o detalle visual.

- No des enlaces.
- No menciones URLs.
- No menciones Wikipedia ni otras webs.
- No uses listas en la respuesta hablada.
- No ofrezcas “versiones largas” ni cierres artificiales.
""".strip()


def build_poi_research_prompt(user_ctx: str, poi_name: str, lat: float, lng: float) -> str:
    return f"""
Necesito memoria interna para un guía turístico presencial.

POI actual: {poi_name}
Ubicación aproximada: latitud {lat}, longitud {lng}
Perfil del viajero: {user_ctx}

Usa búsqueda web para reunir contexto útil y fiable.

Devuelve memoria interna breve y práctica, no texto para leer al visitante.

Incluye:
1. Identificación fiable del lugar.
2. Contexto histórico o cultural básico verificable.
3. Qué elementos visibles puede tener delante el visitante.
4. Posibles confusiones habituales.
5. Preguntas típicas.
6. Qué no debe afirmarse sin confirmación.

Reglas:
- No des enlaces ni URLs.
- No hagas un guion.
- No escribas como si estuvieras hablando al visitante.
- Sé concreto y útil.
""".strip()


def build_turn_prompt(
    user_message: str,
    current_poi_name: str,
    poi_research_summary: str,
    user_ctx: str,
    last_image_summary: str = "",
) -> str:
    return f"""
MENSAJE DEL USUARIO:
{user_message}

POI ACTUAL:
{current_poi_name or "No indicado"}

MEMORIA INTERNA DEL POI:
{poi_research_summary or "No disponible"}

ÚLTIMO CONTEXTO VISUAL:
{last_image_summary or "No disponible"}

PERFIL DEL VIAJERO:
{user_ctx}

Instrucción:
Responde directamente al visitante como Locus, su guía turístico.

Reglas:
- No menciones el proceso interno.
- No menciones contexto, research, fuentes ni notas.
- No hables como asistente externo.
- No conviertas la respuesta en esquema, FAQ o guion.
- Si puedes responder bien con el contexto disponible, hazlo de inmediato.
- Solo si falta un dato factual realmente importante, puedes verificarlo.
- No des enlaces ni URLs.
- No inventes nada.
""".strip()


# ---------------------------------------------------------
# OPENAI HELPERS
# ---------------------------------------------------------

def ask_openai_chat(system_context: str, user_message: str, allow_web_search: bool = False) -> str:
    args = {
        "model": OPENAI_CHAT_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_context}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ],
    }

    if allow_web_search:
        args["tools"] = [{"type": "web_search"}]
        args["tool_choice"] = "auto"

    response = openai_client.responses.create(**args)
    return (response.output_text or "").strip()


def ask_openai_image(
    system_context: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    allow_web_search: bool = False,
) -> str:
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

    args = {
        "model": OPENAI_CHAT_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_context}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            },
        ],
    }

    if allow_web_search:
        args["tools"] = [{"type": "web_search"}]
        args["tool_choice"] = "auto"

    response = openai_client.responses.create(**args)
    return (response.output_text or "").strip()


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = FRONT_AUDIO_SAMPLE_RATE,
    channels: int = FRONT_AUDIO_CHANNELS,
    sample_width: int = FRONT_AUDIO_BYTES_PER_SAMPLE,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def transcribe_pcm16_audio(audio_bytes: bytes, language_hint: Optional[str] = None) -> str:
    wav_bytes = pcm16_to_wav_bytes(audio_bytes)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            kwargs = {
                "model": OPENAI_TRANSCRIBE_MODEL,
                "file": audio_file,
                "response_format": "text",
            }
            if language_hint:
                kwargs["language"] = language_hint

            transcript = openai_client.audio.transcriptions.create(**kwargs)

        if isinstance(transcript, str):
            return transcript.strip()

        return (getattr(transcript, "text", "") or "").strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def build_tts_instructions(user_ctx: str, answer_text: str) -> str:
    lowered = (user_ctx or "").lower()

    language_hint = "Habla en español de España por defecto."
    accent_hint = "Usa un tono natural de guía presencial."
    style_hint = "Voz masculina, natural, segura, cercana y con ritmo ágil."

    if "english" in lowered or "inglés" in lowered or "ingles" in lowered:
        language_hint = "Speak in English."
    elif "italiano" in lowered or "italian" in lowered:
        language_hint = "Parla in italiano."
    elif "francés" in lowered or "frances" in lowered or "french" in lowered:
        language_hint = "Parle en français."

    if "andaluz" in lowered:
        accent_hint = "Usa un color andaluz suave, natural, sin caricatura."
    elif "latino" in lowered or "latam" in lowered:
        accent_hint = "Usa español latino neutro."

    return f"""
{language_hint}
{accent_hint}
{style_hint}
No sobreactúes.
Mantén un ritmo natural de conversación.
""".strip()


def synthesize_speech_wav(text: str, user_ctx: str) -> bytes:
    response = openai_client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        instructions=build_tts_instructions(user_ctx, text),
        response_format="wav",
        speed=1.03,
    )
    return response.read()


def sanitize_for_voice(text: str) -> str:
    if not text:
        return text

    sanitized = re.sub(r'https?://\S+', '', text)
    sanitized = re.sub(r'www\.\S+', '', sanitized)

    replacements = [
        ("Para más información", ""),
        ("para más información", ""),
        ("puedes consultar", ""),
        ("puedes ver", ""),
        ("según Wikipedia", ""),
        ("Wikipedia", ""),
        ("según la web", ""),
        ("en este enlace", ""),
        ("en la siguiente web", ""),
    ]

    for old, new in replacements:
        sanitized = sanitized.replace(old, new)

    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized


def should_force_web_search(user_message: str) -> bool:
    lowered = (user_message or "").lower()

    strict_markers = [
        "fecha exacta",
        "año exacto",
        "autor exacto",
        "modelo exacto",
        "quién lo hizo",
        "quien lo hizo",
        "cuándo fue",
        "cuando fue",
        "estás seguro",
        "estas seguro",
        "seguro?",
        "compruébalo",
        "compruebalo",
        "confírmalo",
        "confirmalo",
        "confírmame",
        "confirmame",
    ]

    return any(marker in lowered for marker in strict_markers)


# ---------------------------------------------------------
# ROOM STATE
# ---------------------------------------------------------

class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.system_context_str = ""
        self.current_poi_name = ""
        self.poi_research_summary = ""
        self.last_image_summary = ""
        self.poi_research_task: Optional[asyncio.Task] = None

        self.profile = {
            "user_ctx": "",
            "lat": 0.0,
            "lng": 0.0,
            "pois_data": "[]",
        }


# ---------------------------------------------------------
# CONNECTION MANAGER
# ---------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.room_states: dict[str, RoomState] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()

        if room_id not in self.active_connections:
            self.active_connections[room_id] = []

        if room_id not in self.room_states:
            self.room_states[room_id] = RoomState(room_id)

        self.active_connections[room_id].append(websocket)
        logger.info(f"[{room_id}] Cliente conectado. Total={len(self.active_connections[room_id])}")

    async def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections and websocket in self.active_connections[room_id]:
            self.active_connections[room_id].remove(websocket)

        if room_id in self.active_connections and not self.active_connections[room_id]:
            logger.info(f"[{room_id}] Último cliente desconectado. Cerrando sala.")
            del self.active_connections[room_id]

            room_state = self.room_states.pop(room_id, None)
            if room_state and room_state.poi_research_task and not room_state.poi_research_task.done():
                room_state.poi_research_task.cancel()

    async def broadcast_text(self, text: str, room_id: str):
        if room_id not in self.active_connections:
            return

        stale = []
        for connection in self.active_connections[room_id]:
            try:
                await connection.send_text(text)
            except Exception:
                stale.append(connection)

        for connection in stale:
            try:
                self.active_connections[room_id].remove(connection)
            except ValueError:
                pass

    async def broadcast_bytes(self, data: bytes, room_id: str):
        if room_id not in self.active_connections:
            return

        stale = []
        for connection in self.active_connections[room_id]:
            try:
                await connection.send_bytes(data)
            except Exception:
                stale.append(connection)

        for connection in stale:
            try:
                self.active_connections[room_id].remove(connection)
            except ValueError:
                pass


manager = ConnectionManager()


# ---------------------------------------------------------
# BACKGROUND ENRICHMENT
# ---------------------------------------------------------

async def enrich_poi_context_in_background(room_state: RoomState):
    try:
        poi_name = room_state.current_poi_name or "el lugar actual"
        lat = float(room_state.profile.get("lat", 0.0))
        lng = float(room_state.profile.get("lng", 0.0))
        user_ctx = room_state.profile.get("user_ctx", "")

        system_context = """
Eres un asistente de documentación para un guía turístico.
Tu trabajo es preparar memoria interna fiable y breve.
No pongas enlaces ni URLs.
No inventes nada.
""".strip()

        prompt = build_poi_research_prompt(user_ctx, poi_name, lat, lng)

        summary = await asyncio.to_thread(
            ask_openai_chat,
            system_context,
            prompt,
            True,
        )

        room_state.poi_research_summary = summary.strip()
        logger.info(f"[{room_state.room_id}] Dossier del POI enriquecido")

    except asyncio.CancelledError:
        logger.info(f"[{room_state.room_id}] Enriquecimiento cancelado")
        raise
    except Exception as e:
        logger.exception(f"[{room_state.room_id}] Error enriqueciendo POI: {e}")


# ---------------------------------------------------------
# CALL LOGIC
# ---------------------------------------------------------

async def speak_text(room_state: RoomState, room_id: str, text: str):
    clean_text = sanitize_for_voice(text)
    if not clean_text:
        return

    await manager.broadcast_text(clean_text, room_id)

    try:
        audio_bytes = await asyncio.to_thread(
            synthesize_speech_wav,
            clean_text,
            room_state.profile.get("user_ctx", ""),
        )
        if audio_bytes:
            await manager.broadcast_bytes(audio_bytes, room_id)
    except Exception as e:
        logger.exception(f"[{room_id}] Error generando TTS: {e}")


async def generate_fast_answer(room_state: RoomState, user_message: str) -> str:
    system_context = build_voice_context(
        user_ctx=room_state.profile.get("user_ctx", ""),
        lat=float(room_state.profile.get("lat", 0.0)),
        lng=float(room_state.profile.get("lng", 0.0)),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=room_state.current_poi_name,
        poi_research_summary=room_state.poi_research_summary,
        last_image_summary=room_state.last_image_summary,
    )

    prompt = build_turn_prompt(
        user_message=user_message,
        current_poi_name=room_state.current_poi_name,
        poi_research_summary=room_state.poi_research_summary,
        user_ctx=room_state.profile.get("user_ctx", ""),
        last_image_summary=room_state.last_image_summary,
    )

    return await asyncio.to_thread(
        ask_openai_chat,
        system_context,
        prompt,
        False,
    )


async def generate_verified_answer(room_state: RoomState, user_message: str) -> str:
    system_context = build_voice_context(
        user_ctx=room_state.profile.get("user_ctx", ""),
        lat=float(room_state.profile.get("lat", 0.0)),
        lng=float(room_state.profile.get("lng", 0.0)),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=room_state.current_poi_name,
        poi_research_summary=room_state.poi_research_summary,
        last_image_summary=room_state.last_image_summary,
    )

    prompt = build_turn_prompt(
        user_message=user_message,
        current_poi_name=room_state.current_poi_name,
        poi_research_summary=room_state.poi_research_summary,
        user_ctx=room_state.profile.get("user_ctx", ""),
        last_image_summary=room_state.last_image_summary,
    )

    return await asyncio.to_thread(
        ask_openai_chat,
        system_context,
        prompt,
        True,
    )


async def respond_in_call(room_state: RoomState, room_id: str, user_message: str):
    logger.info(f"[{room_id}] Pregunta call -> {user_message[:180]}")

    answer = await generate_fast_answer(room_state, user_message)

    if not answer:
        answer = "No lo tengo claro del todo. Cuéntame qué ves exactamente y lo afinamos."

    await speak_text(room_state, room_id, answer)

    if should_force_web_search(user_message):
        async def refine_later():
            try:
                better_answer = await generate_verified_answer(room_state, user_message)
                better_answer = sanitize_for_voice(better_answer)

                if better_answer and better_answer != sanitize_for_voice(answer):
                    logger.info(f"[{room_id}] Respuesta refinada disponible")
                    await manager.broadcast_text(f"ℹ️ {better_answer}", room_id)
            except Exception as e:
                logger.exception(f"[{room_id}] Error refinando respuesta: {e}")

        asyncio.create_task(refine_later())


async def respond_to_image_in_call(
    room_state: RoomState,
    room_id: str,
    image_bytes: bytes,
    mime_type: str,
):
    poi_name = room_state.current_poi_name or "el lugar actual"

    system_context = build_voice_context(
        user_ctx=room_state.profile.get("user_ctx", ""),
        lat=float(room_state.profile.get("lat", 0.0)),
        lng=float(room_state.profile.get("lng", 0.0)),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=room_state.current_poi_name,
        poi_research_summary=room_state.poi_research_summary,
        last_image_summary=room_state.last_image_summary,
    )

    prompt = (
        f"El usuario está visitando {poi_name}. "
        "Analiza esta imagen como si fueras su guía turístico. "
        "Identifica la obra, sala, objeto o elemento principal si puedes hacerlo con base suficiente. "
        "Usa búsqueda web si hace falta. "
        "No inventes nada. "
        "No des enlaces ni menciones URLs. "
        "Si no puedes identificarlo con seguridad, dilo claramente y explica solo lo que puedas sostener. "
        "Si puedes identificarlo, cuéntalo de forma útil para una visita real."
    )

    answer = await asyncio.to_thread(
        ask_openai_image,
        system_context,
        prompt,
        image_bytes,
        mime_type,
        True,
    )

    if not answer:
        answer = "No consigo identificarlo con seguridad. Si quieres, acércame la placa o encuadra mejor el detalle principal."

    room_state.last_image_summary = answer
    await speak_text(room_state, room_id, answer)


def infer_language_hint(user_ctx: str) -> Optional[str]:
    lowered = (user_ctx or "").lower()

    if "english" in lowered or "inglés" in lowered or "ingles" in lowered:
        return "en"
    if "italiano" in lowered or "italian" in lowered:
        return "it"
    if "francés" in lowered or "frances" in lowered or "french" in lowered:
        return "fr"

    return "es"


# ---------------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------------

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, deviceId: str = Query(None)):
    await manager.connect(websocket, room_id)
    room_state = manager.room_states[room_id]

    try:
        while True:
            msg = await websocket.receive()

            if "text" not in msg:
                continue

            raw_data = msg["text"]
            if not raw_data:
                continue

            if raw_data.strip().startswith("{"):
                try:
                    payload = json.loads(raw_data)
                    action = payload.get("action")

                    if action == "setup_profile":
                        user_ctx = payload.get("context", "")
                        lat = float(payload.get("lat", 0.0))
                        lng = float(payload.get("lng", 0.0))
                        pois_data = buscar_nuevo_lugar("atracciones turisticas", lat, lng)

                        room_state.profile = {
                            "user_ctx": user_ctx,
                            "lat": lat,
                            "lng": lng,
                            "pois_data": pois_data,
                        }

                        room_state.system_context_str = build_home_context(
                            user_ctx=user_ctx,
                            lat=lat,
                            lng=lng,
                            pois_data=pois_data,
                        )

                        response_text = await asyncio.to_thread(
                            ask_openai_chat,
                            room_state.system_context_str,
                            "Hola, acabo de llegar. Preséntate como guía y dime qué hay cerca. Incluye los POIs en el formato obligatorio.",
                            False,
                        )

                        await manager.broadcast_text(response_text, room_id)
                        continue

                    if action == "start_voice_call":
                        room_state.current_poi_name = payload.get("poi_name", "tu destino")
                        room_state.last_image_summary = ""

                        if room_state.poi_research_task and not room_state.poi_research_task.done():
                            room_state.poi_research_task.cancel()

                        room_state.poi_research_task = asyncio.create_task(
                            enrich_poi_context_in_background(room_state)
                        )

                        await respond_in_call(
                            room_state,
                            room_id,
                            (
                                f"El usuario acaba de iniciar la ruta en {room_state.current_poi_name}. "
                                "Dale la bienvenida de forma breve, recomienda ponerse auriculares si hay ruido y pregunta qué tiene delante."
                            ),
                        )
                        continue

                    if action == "text_chat":
                        user_text = (payload.get("data") or "").strip()
                        if user_text:
                            await respond_in_call(room_state, room_id, user_text)
                        continue

                    if action == "audio_stream":
                        continue

                    if action == "audio_end":
                        continue

                    if action == "audio_chat":
                        try:
                            audio_b64 = payload.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)

                            if not audio_bytes:
                                continue

                            language_hint = infer_language_hint(room_state.profile.get("user_ctx", ""))

                            transcript = await asyncio.to_thread(
                                transcribe_pcm16_audio,
                                audio_bytes,
                                language_hint,
                            )

                            if not transcript:
                                await manager.broadcast_text(
                                    "No te he entendido bien. Repite la pregunta si quieres.",
                                    room_id,
                                )
                                continue

                            logger.info(f"[{room_id}] Transcripción -> {transcript}")
                            await respond_in_call(room_state, room_id, transcript)

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando audio_chat: {e}")
                            await manager.broadcast_text(
                                "Ha habido un problema procesando el audio.",
                                room_id,
                            )
                        continue

                    if action == "image_context":
                        try:
                            img_b64 = payload.get("data", "")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            img_bytes = base64.b64decode(img_b64)

                            if not img_bytes:
                                continue

                            await respond_to_image_in_call(
                                room_state,
                                room_id,
                                img_bytes,
                                mime_type,
                            )

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando image_context: {e}")
                            await manager.broadcast_text(
                                "Ha habido un problema procesando la imagen.",
                                room_id,
                            )
                        continue

                except Exception as e:
                    logger.exception(f"[{room_id}] Error procesando JSON: {e}")
                    continue

            try:
                response_text = await asyncio.to_thread(
                    ask_openai_chat,
                    room_state.system_context_str,
                    raw_data,
                    False,
                )
                await manager.broadcast_text(response_text, room_id)
            except Exception as e:
                logger.exception(f"[{room_id}] Error en chat OpenAI: {e}")
                await manager.broadcast_text(
                    "No he podido procesar ese mensaje.",
                    room_id,
                )

    except WebSocketDisconnect:
        await manager.disconnect(websocket, room_id)
    except Exception as e:
        logger.exception(f"[{room_id}] WebSocket Error: {e}")
        await manager.disconnect(websocket, room_id)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)