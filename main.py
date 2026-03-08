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
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "onyx")
OPENAI_TRANSCRIBE_MODEL = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "whisper-1")

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
                            "description": "Lugar cercano",
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
    return f"""Eres Locus, un guía turístico local, cercano, resolutivo y con personalidad.

CONTEXTO DEL VIAJERO:
{user_ctx}

UBICACIÓN ACTUAL:
Latitud {lat}
Longitud {lng}

LUGARES CERCANOS DETECTADOS:
{pois_data}

REGLAS:
1. Habla como guía turístico real, no como asistente genérico.
2. Adapta idioma, energía y forma de hablar al perfil del viajero.
3. Si el viajero pregunta qué hay cerca o acaba de llegar, incluye POIs.
4. Si incluyes POIs, dilo de forma natural para que mire el mapa.
5. No inventes datos concretos.
6. No des enlaces ni URLs.
7. El formato de POIs es estrictamente este:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
"""


def build_voice_context(
    user_ctx: str,
    pois_data: str,
    poi_name: str,
    research: str,
    last_img: str
) -> str:
    return f"""Eres Locus, un guía turístico presencial, masculino, natural, observador y con criterio. Estás acompañando de verdad a estos viajeros durante una visita real.

PERFIL DE LOS VIAJEROS:
{user_ctx}

POI ACTUAL DE LA VISITA:
{poi_name if poi_name else "NO DEFINIDO"}

POIS CERCANOS:
{pois_data}

NOTAS INTERNAS DE APOYO SOBRE EL POI ACTUAL:
{research if research else "Aún no hay notas enriquecidas."}

ÚLTIMO DETALLE VISUAL COMENTADO:
{last_img if last_img else "Ninguno."}

REGLA CRÍTICA ABSOLUTA:
Habla SIEMPRE de forma directa con el usuario.
NO uses asteriscos.
NO pienses en voz alta.
NO expliques tu proceso.
NO generes monólogos internos.
NO hables como asistente.
Di únicamente el texto final que el usuario va a escuchar de tu voz.

REGLA CRÍTICA DE FOCO:
El lugar actual de la visita es {poi_name if poi_name else "el POI actual"}.
Mientras el usuario no diga claramente que se ha movido, asume SIEMPRE que sigue en ese mismo sitio.
NO le preguntes otra vez en qué lugar está.
NO cambies el foco a otros POIs cercanos.
Si el usuario pregunta algo ambiguo, interprétalo siempre dentro del POI actual.

REGLA CRÍTICA DE VERACIDAD:
Nunca inventes una identificación concreta de un objeto, obra, locomotora, estatua o elemento arquitectónico.
Nunca uses “probablemente”, “quizá”, “seguramente” o suposiciones flojas como si fueran datos.
Si no puedes identificar algo con seguridad, dilo claramente y pide el detalle mínimo útil para afinar, como una placa, una inscripción, un número, una foto más cercana o un ángulo mejor.

ESTILO DE LOCUS:
- Suenas como un guía bueno de verdad, no como un folleto ni como una IA blandita.
- Tienes iniciativa.
- Vas al detalle.
- Si el usuario pide historia, curiosidades, modelo, contexto o profundidad, entras en modo experto y profundizas de verdad.
- Si el usuario solo hace una pregunta breve, puedes responder breve.
- No repitas el contexto general del lugar si ya estáis hablando de un objeto concreto.

REGLAS DE RESPUESTA:
1. Si el usuario señala un objeto concreto, habla de ese objeto, no del lugar en general.
2. Si el usuario pide profundidad, detalle, historia, curiosidades o contexto técnico, responde con más sustancia y baja al detalle.
3. Si ya has presentado el lugar, no vuelvas a presentarlo. Avanza.
4. Si puedes sostener un dato con seguridad, dilo con convicción.
5. Si no puedes sostenerlo, sé honesto.
6. No cierres siempre con una pregunta vacía. Solo pregunta algo si de verdad ayuda a continuar la visita.
7. No des enlaces ni menciones webs.
"""


def build_poi_research_prompt(user_ctx: str, poi_name: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""Eres el documentalista de un guía turístico experto.

Tu misión es preparar notas internas útiles y fiables sobre este lugar:
{poi_name}

Contexto:
- Perfil del viajero: {user_ctx}
- Coordenadas aproximadas: lat {lat}, lng {lng}
- Zona cercana / POIs próximos: {pois_data}

Devuelve solo notas internas, en viñetas rápidas y concretas:
- Qué es exactamente el lugar
- Fecha, época o etapa histórica relevante si se conoce con seguridad
- Autor, promotor o institución responsable si aplica
- 2 o 3 datos históricos o culturales potentes
- 2 detalles visuales útiles para una visita en persona
- 1 confusión habitual o error común que conviene evitar
- Si el lugar contiene un objeto emblemático conocido, menciónalo

Reglas:
- Sin relleno
- Sin introducciones
- Sin tono literario
- Sin enlaces ni URLs
- No inventes nada
"""


def build_history_text(chat_history: list[dict]) -> str:
    lines = []
    for msg in chat_history[-12:]:
        role = "Usuario" if msg["role"] == "user" else "Locus"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def detect_depth_mode(user_message: str) -> bool:
    text = (user_message or "").lower()
    triggers = [
        "más", "mas", "profund", "detalle", "detalles", "historia",
        "curiosidad", "curiosidades", "modelo", "por qué", "porque",
        "explícame", "explicame", "en profundidad", "más a fondo",
        "cuéntame", "cuentame", "quiero saber más", "quiero saber mas",
        "toda la historia", "contexto", "origen"
    ]
    return any(t in text for t in triggers)


# ---------------------------------------------------------
# OPENAI HELPERS
# ---------------------------------------------------------

def ask_openai_chat(system_context: str, user_message: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def ask_openai_image(
    system_context: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str
) -> str:
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_context},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def ask_openai_research(system_context: str, user_message: str) -> str:
    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        tools=[{"type": "web_search_preview"}],
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_context}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ],
    )
    return (response.output_text or "").strip()


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = FRONT_AUDIO_SAMPLE_RATE,
    channels: int = FRONT_AUDIO_CHANNELS,
    sample_width: int = FRONT_AUDIO_BYTES_PER_SAMPLE
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


def synthesize_speech_wav(text: str) -> bytes:
    response = openai_client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="wav",
        speed=1.08,
    )
    return response.read()


def sanitize_for_voice(text: str) -> str:
    if not text:
        return text

    sanitized = re.sub(r'https?://\S+', '', text)
    sanitized = re.sub(r'www\.\S+', '', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized


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
# ROOM STATE
# ---------------------------------------------------------

class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.system_context_str = ""

        self.profile = {
            "user_ctx": "",
            "lat": 0.0,
            "lng": 0.0,
            "pois_data": "[]",
        }

        self.current_poi_name = ""
        self.poi_research_summary = ""
        self.last_image_summary = ""
        self.chat_history: list[dict] = []

        self.poi_research_task: Optional[asyncio.Task] = None

        self.turn_lock = asyncio.Lock()
        self.call_start_lock = asyncio.Lock()

        self.call_active = False
        self.poi_generation = 0
        self.welcome_sent_for_generation = False


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

async def enrich_poi_context_in_background(room_state: RoomState, poi_generation: int, poi_name: str):
    try:
        lat = float(room_state.profile.get("lat", 0.0))
        lng = float(room_state.profile.get("lng", 0.0))
        user_ctx = room_state.profile.get("user_ctx", "")
        pois_data = room_state.profile.get("pois_data", "[]")

        prompt = build_poi_research_prompt(user_ctx, poi_name, lat, lng, pois_data)

        summary = await asyncio.to_thread(
            ask_openai_research,
            "Eres un documentalista preciso. Ve al grano. No inventes nada. Sin enlaces.",
            prompt
        )

        if room_state.poi_generation != poi_generation:
            logger.info(f"[{room_state.room_id}] Research obsoleto descartado para {poi_name}")
            return

        if room_state.current_poi_name != poi_name:
            logger.info(f"[{room_state.room_id}] Research descartado porque el POI cambió")
            return

        room_state.poi_research_summary = summary.strip()
        logger.info(f"[{room_state.room_id}] Contexto enriquecido para {poi_name}")

    except asyncio.CancelledError:
        logger.info(f"[{room_state.room_id}] Enriquecimiento cancelado")
        raise
    except Exception as e:
        logger.exception(f"[{room_state.room_id}] Error enriqueciendo POI: {e}")


# ---------------------------------------------------------
# GENERACIÓN DE RESPUESTAS
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
        )
        if audio_bytes:
            await manager.broadcast_bytes(audio_bytes, room_id)
    except Exception as e:
        logger.exception(f"[{room_id}] Error generando TTS: {e}")


async def generate_fast_answer(room_state: RoomState, user_message: str) -> str:
    system_context = build_voice_context(
        user_ctx=room_state.profile.get("user_ctx", ""),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=room_state.current_poi_name,
        research=room_state.poi_research_summary,
        last_img=room_state.last_image_summary,
    )

    history_text = build_history_text(room_state.chat_history)
    depth_mode = detect_depth_mode(user_message)

    answer_style = (
        "MODO EXPERTO: el usuario ha pedido más profundidad. Da una respuesta más rica, concreta y útil. "
        "No te quedes en generalidades. Si hablas de un objeto concreto, entra en el detalle histórico o técnico real que puedas sostener."
        if depth_mode
        else
        "MODO NORMAL: responde de forma natural, útil y directa."
    )

    prompt = f"""HISTORIAL RECIENTE DE LA VISITA:
{history_text}

MENSAJE NUEVO DEL USUARIO:
{user_message}

{answer_style}

RESPONDE COMO LOCUS:"""

    answer = await asyncio.to_thread(
        ask_openai_chat,
        system_context,
        prompt
    )

    room_state.chat_history.append({"role": "user", "content": user_message})
    room_state.chat_history.append({"role": "assistant", "content": answer})
    room_state.chat_history = room_state.chat_history[-12:]

    return answer


async def respond_in_call(room_state: RoomState, room_id: str, user_message: str):
    async with room_state.turn_lock:
        answer = await generate_fast_answer(room_state, user_message)
        if not answer:
            answer = "No te he entendido bien. Dímelo otra vez y voy al grano."
        await speak_text(room_state, room_id, answer)


async def respond_to_image_in_call(room_state: RoomState, room_id: str, image_bytes: bytes, mime_type: str):
    async with room_state.turn_lock:
        system_context = build_voice_context(
            user_ctx=room_state.profile.get("user_ctx", ""),
            pois_data=room_state.profile.get("pois_data", "[]"),
            poi_name=room_state.current_poi_name,
            research=room_state.poi_research_summary,
            last_img="",
        )

        history_text = build_history_text(room_state.chat_history)

        prompt = f"""HISTORIAL RECIENTE DE LA VISITA:
{history_text}

El usuario te está enseñando un detalle que tiene ahora mismo delante en {room_state.current_poi_name}.
Identifica con seguridad lo que puedas.
Si no puedes identificarlo con seguridad, dilo claramente y describe solo lo que sí puedes sostener.
Habla como su guía turístico presencial, no como una herramienta de visión artificial.

RESPONDE COMO LOCUS:"""

        answer = await asyncio.to_thread(
            ask_openai_image,
            system_context,
            prompt,
            image_bytes,
            mime_type
        )

        if not answer:
            answer = "Así no lo veo del todo claro. Acércame un poco más el detalle o la placa y lo afinamos."

        room_state.last_image_summary = answer
        room_state.chat_history.append({"role": "user", "content": "[El usuario ha enseñado una imagen del detalle que tiene delante]"})
        room_state.chat_history.append({"role": "assistant", "content": answer})
        room_state.chat_history = room_state.chat_history[-12:]

        await speak_text(room_state, room_id, answer)


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
                            "Hola, acabamos de llegar. Preséntate como guía, da la bienvenida y dinos qué hay cerca. Si muestras POIs, recuerda decirnos que miremos el mapa."
                        )
                        await manager.broadcast_text(response_text, room_id)
                        continue

                    if action == "start_voice_call":
                        requested_poi = payload.get("poi_name", "tu destino").strip()

                        async with room_state.call_start_lock:
                            if room_state.call_active and room_state.current_poi_name == requested_poi:
                                logger.info(f"[{room_id}] start_voice_call duplicado ignorado para {requested_poi}")
                                continue

                            room_state.current_poi_name = requested_poi
                            room_state.last_image_summary = ""
                            room_state.chat_history = []
                            room_state.poi_research_summary = ""
                            room_state.call_active = True
                            room_state.poi_generation += 1
                            room_state.welcome_sent_for_generation = False
                            current_generation = room_state.poi_generation

                            if room_state.poi_research_task and not room_state.poi_research_task.done():
                                room_state.poi_research_task.cancel()

                            room_state.poi_research_task = asyncio.create_task(
                                enrich_poi_context_in_background(
                                    room_state,
                                    current_generation,
                                    requested_poi
                                )
                            )

                            if not room_state.welcome_sent_for_generation:
                                room_state.welcome_sent_for_generation = True
                                await respond_in_call(
                                    room_state,
                                    room_id,
                                    f"El usuario acaba de iniciar la llamada y ha seleccionado {room_state.current_poi_name}. Saluda una sola vez, recomienda cascos si hay ruido y pregunta si empezamos."
                                )
                        continue

                    if action == "text_chat":
                        user_text = (payload.get("data") or "").strip()
                        if user_text:
                            await respond_in_call(room_state, room_id, user_text)
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
                                continue

                            await respond_in_call(room_state, room_id, transcript)

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando audio_chat: {e}")
                        continue

                    if action == "image_context":
                        try:
                            img_b64 = payload.get("data", "")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            img_bytes = base64.b64decode(img_b64)
                            if not img_bytes:
                                continue

                            await respond_to_image_in_call(room_state, room_id, img_bytes, mime_type)

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando image_context: {e}")
                        continue

                except Exception as e:
                    logger.exception(f"[{room_id}] Error procesando JSON: {e}")
                    continue

            try:
                response_text = await asyncio.to_thread(
                    ask_openai_chat,
                    room_state.system_context_str,
                    raw_data
                )
                await manager.broadcast_text(response_text, room_id)
            except Exception as e:
                logger.exception(f"[{room_id}] Error en chat OpenAI: {e}")

    except WebSocketDisconnect:
        await manager.disconnect(websocket, room_id)
    except Exception as e:
        logger.exception(f"[{room_id}] WebSocket Error: {e}")
        await manager.disconnect(websocket, room_id)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)