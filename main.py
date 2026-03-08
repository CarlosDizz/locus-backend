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
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
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
1. Habla siempre como guía turístico real, no como asistente genérico.
2. Adapta tu idioma, energía y forma de hablar al perfil del viajero.
3. Si el viajero acaba de llegar o pregunta qué hay cerca, debes incluir POIs.
4. Cuando muestres POIs, díselo al usuario de forma natural para que mire el mapa.
5. No inventes datos concretos.
6. No des la razón por darla: di la verdad técnica.
7. No des enlaces ni URLs.
8. El formato de POIs es estrictamente este:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
"""


def build_voice_context(
    user_ctx: str,
    lat: float,
    lng: float,
    pois_data: str,
    poi_name: str,
    research: str,
    last_img: str
) -> str:
    ubicacion_foco = f"ESTÁS FÍSICAMENTE EN: {poi_name}." if poi_name else "Caminando por la ciudad, sin destino fijo."

    info = ""
    if research:
        info += f"\nTUS NOTAS DE EXPERTO (NO LAS LEAS LITERALMENTE, ÚSALAS PARA HABLAR MEJOR):\n{research}\n"
    if last_img:
        info += f"\nÚLTIMO DETALLE VISUAL DEL QUE ESTÁ HABLANDO EL USUARIO:\n{last_img}\n"

    return f"""Eres Locus, un guía turístico presencial, carismático, observador y natural. Estás acompañando de verdad a estos viajeros durante la visita.

PERFIL DE LOS VIAJEROS:
{user_ctx}

{ubicacion_foco}

LUGARES CERCANOS:
{pois_data}
{info}

REGLA CRÍTICA ABSOLUTA:
Habla SIEMPRE de forma directa con el usuario.
NO uses asteriscos.
NO pienses en voz alta.
NO expliques tu proceso.
NO generes monólogos internos.
NO hables como asistente.
Di únicamente el texto final que el usuario va a escuchar de tu voz.

REGLAS DE COMPORTAMIENTO:
1. MEMORIA Y CONTINUIDAD: Si el usuario habla de un detalle concreto (una locomotora, una puerta, una estatua, una pintura), mantén el foco ahí. No vuelvas al contexto general salvo que sea necesario.
2. PROFUNDIDAD: No repitas datos ya dichos. Si ya presentaste el lugar, baja al detalle: un secreto, una anécdota, un rasgo arquitectónico, un fallo curioso o una lectura más fina.
3. ADAPTACIÓN: Ajusta idioma, energía, tono y acento al perfil del usuario. Nunca pidas confirmar en qué ciudad estáis.
4. BREVEDAD: Responde en 2 o 3 frases como máximo, salvo que el usuario pida claramente más detalle.
5. VERDAD: No inventes nada. Si no estás seguro, dilo con naturalidad.
6. NATURALIDAD: Suena como un guía real, no como una app ni como un documental.
7. CIERRE: Mantén viva la visita y termina de forma conversacional, preferiblemente devolviendo la atención a lo que están viendo.
"""


def build_poi_research_prompt(user_ctx: str, poi_name: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""Eres el documentalista de un guía turístico experto.

TU MISIÓN:
Preparar notas internas muy útiles sobre este lugar: {poi_name}

CONTEXTO:
- Perfil de la audiencia: {user_ctx}
- Coordenadas aproximadas: lat {lat}, lng {lng}
- Zona cercana / POIs próximos: {pois_data}

DEVUELVE SOLO NOTAS INTERNAS, EN VIÑETAS RÁPIDAS Y CONCRETAS:
- Identificación fiable del lugar
- Año o época de creación si se conoce con seguridad
- Autor, impulsor o institución responsable si se conoce
- 1 anécdota potente o dato poco obvio
- 2 detalles visuales en los que fijarse in situ
- 1 posible confusión habitual o error común que conviene no cometer

REGLAS:
- Sin relleno
- Sin introducciones
- Sin tono literario
- Sin enlaces ni URLs
- No inventes nada
"""


# ---------------------------------------------------------
# OPENAI HELPERS
# ---------------------------------------------------------

def ask_openai_chat(system_context: str, user_message: str, history: list | None = None) -> str:
    messages = [{"role": "system", "content": system_context}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def ask_openai_image(
    system_context: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    history: list | None = None
) -> str:
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

    messages = [{"role": "system", "content": system_context}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    })

    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def ask_openai_research(system_context: str, user_message: str) -> str:
    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        tools=[{"type": "web_search"}],
        tool_choice="auto",
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
        speed=1.15,
    )
    return response.read()


def sanitize_for_voice(text: str) -> str:
    if not text:
        return text

    sanitized = re.sub(r'https?://\S+', '', text)
    sanitized = re.sub(r'www\.\S+', '', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized


def build_history_text(chat_history: list[dict]) -> str:
    lines = []
    for msg in chat_history[-10:]:
        role = "Usuario" if msg["role"] == "user" else "Locus"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


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
        self.chat_history: list[dict] = []
        self.poi_research_task: Optional[asyncio.Task] = None

        self.profile = {
            "user_ctx": "",
            "lat": 0.0,
            "lng": 0.0,
            "pois_data": "[]",
        }

        self.turn_lock = asyncio.Lock()
        self.call_active = False
        self.poi_generation = 0
        self.last_public_text = ""


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

    async def send_text_to_one(self, websocket: WebSocket, text: str):
        try:
            await websocket.send_text(text)
        except Exception:
            pass

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
            "Eres un documentalista de élite. Ve al grano. No inventes nada. No des enlaces ni URLs.",
            prompt
        )

        if room_state.poi_generation != poi_generation:
            logger.info(f"[{room_state.room_id}] Research obsoleto descartado para {poi_name}")
            return

        if room_state.current_poi_name != poi_name:
            logger.info(f"[{room_state.room_id}] Research descartado: POI actual cambió")
            return

        room_state.poi_research_summary = summary.strip()
        logger.info(f"[{room_state.room_id}] Contexto enriquecido para {poi_name}")

    except asyncio.CancelledError:
        logger.info(f"[{room_state.room_id}] Enriquecimiento cancelado")
        raise
    except Exception as e:
        logger.exception(f"[{room_state.room_id}] Error enriqueciendo POI: {e}")


# ---------------------------------------------------------
# SPEAK / ANSWERS
# ---------------------------------------------------------

async def speak_text(room_state: RoomState, room_id: str, text: str):
    clean_text = sanitize_for_voice(text)
    if not clean_text:
        return

    room_state.last_public_text = clean_text
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
        lat=float(room_state.profile.get("lat", 0.0)),
        lng=float(room_state.profile.get("lng", 0.0)),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=room_state.current_poi_name,
        research=room_state.poi_research_summary,
        last_img=room_state.last_image_summary
    )

    history_text = build_history_text(room_state.chat_history)

    prompt = f"""HISTORIAL RECIENTE DE LA VISITA:
{history_text}

NUEVO MENSAJE DEL USUARIO:
{user_message}

RESPONDE COMO LOCUS:"""

    answer = await asyncio.to_thread(
        ask_openai_chat,
        system_context,
        prompt,
        None
    )

    room_state.chat_history.append({"role": "user", "content": user_message})
    room_state.chat_history.append({"role": "assistant", "content": answer})
    room_state.chat_history = room_state.chat_history[-12:]

    return answer


async def respond_in_call(room_state: RoomState, room_id: str, user_message: str):
    async with room_state.turn_lock:
        answer = await generate_fast_answer(room_state, user_message)
        if not answer:
            answer = "No te he escuchado bien, ¿qué decías exactamente de esto?"
        await speak_text(room_state, room_id, answer)


async def respond_to_image_in_call(room_state: RoomState, room_id: str, image_bytes: bytes, mime_type: str):
    async with room_state.turn_lock:
        poi_name = room_state.current_poi_name or "el lugar actual"

        system_context = build_voice_context(
            user_ctx=room_state.profile.get("user_ctx", ""),
            lat=float(room_state.profile.get("lat", 0.0)),
            lng=float(room_state.profile.get("lng", 0.0)),
            pois_data=room_state.profile.get("pois_data", "[]"),
            poi_name=poi_name,
            research=room_state.poi_research_summary,
            last_img=""
        )

        history_text = build_history_text(room_state.chat_history)

        prompt = f"""HISTORIAL RECIENTE DE LA VISITA:
{history_text}

TE ESTOY ENSEÑANDO ESTO QUE ESTOY VIENDO AHORA MISMO EN {poi_name}.
Tenlo en cuenta para la interacción y explícame qué es como mi guía turístico.
Si puedes identificarlo con seguridad, dilo.
Si no puedes identificarlo con seguridad, sé honesto y describe solo lo que sí puedes sostener.

RESPONDE COMO LOCUS:"""

        answer = await asyncio.to_thread(
            ask_openai_image,
            system_context,
            prompt,
            image_bytes,
            mime_type,
            None
        )

        if not answer:
            answer = "No lo veo del todo claro así. Acércame un poco más el detalle o la placa y lo afinamos."

        room_state.last_image_summary = answer

        room_state.chat_history.append({"role": "user", "content": "[El usuario me ha enseñado una imagen del detalle que tiene delante]"})
        room_state.chat_history.append({"role": "assistant", "content": answer})
        room_state.chat_history = room_state.chat_history[-12:]

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
                            "Hola, acabamos de llegar. Preséntate como guía, da la bienvenida y dinos qué hay cerca. Si muestras POIs, recuerda decirnos que miremos el mapa."
                        )
                        await manager.broadcast_text(response_text, room_id)
                        continue

                    if action == "start_voice_call":
                        requested_poi = payload.get("poi_name", "tu destino").strip()

                        if room_state.call_active and room_state.current_poi_name == requested_poi:
                            logger.info(f"[{room_id}] start_voice_call duplicado ignorado para {requested_poi}")
                            continue

                        room_state.current_poi_name = requested_poi
                        room_state.last_image_summary = ""
                        room_state.chat_history = []
                        room_state.poi_research_summary = ""
                        room_state.call_active = True
                        room_state.poi_generation += 1
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

                        await respond_in_call(
                            room_state,
                            room_id,
                            f"El usuario acaba de iniciar la llamada y ha seleccionado {room_state.current_poi_name}. Salúdale por voz, recomienda cascos si hay ruido y pregúntale si empezamos."
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