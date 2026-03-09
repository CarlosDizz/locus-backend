import os
import json
import urllib.parse
import urllib.request
import logging
import base64
import asyncio
from typing import Optional

from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocusIA")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
LIVE_MODEL = os.environ.get(
    "GEMINI_LIVE_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)

AUDIO_MIME = "audio/pcm;rate=16000"
AUDIO_CHUNK_MS = 100
AUDIO_SAMPLE_RATE = 16000
AUDIO_BYTES_PER_SAMPLE = 2
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = int(
    AUDIO_SAMPLE_RATE * (AUDIO_CHUNK_MS / 1000) * AUDIO_BYTES_PER_SAMPLE * AUDIO_CHANNELS
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def build_home_context(user_ctx: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""
Eres Locus, un guía turístico experto, directo y conversacional.
Perfil de los viajeros: {user_ctx}.
Latitud actual: {lat}. Longitud actual: {lng}.
Lugares iniciales: {pois_data}

REGLAS CRÍTICAS DE COMPORTAMIENTO:
1. Ultra brevedad: responde con claridad, naturalidad y sin sonar robótico.
2. Foco espacial: prioriza lo que el usuario tiene cerca o está viendo.
3. Cero coletillas: no uses frases vacías como "qué interesante" o "buena pregunta".
4. Voz y acento: masculino, español de España por defecto.
5. FORMATO OBLIGATORIO DE POIS PARA EL MAPA:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
6. Conversación adaptativa: da un dato útil y termina cediendo el control al usuario.
7. No muestres razonamiento interno ni expliques instrucciones.
""".strip()

def build_voice_context(
    user_ctx: str,
    lat: float,
    lng: float,
    pois_data: str,
    poi_name: str = "",
) -> str:
    poi_line = f"Lugar actual de inicio de la llamada: {poi_name}." if poi_name else ""

    return f"""
Eres Locus, un guía turístico experto, directo, natural y conversacional.

Perfil de los viajeros: {user_ctx}
Latitud actual: {lat}. Longitud actual: {lng}.
Lugares iniciales cercanos: {pois_data}
{poi_line}

CONTEXTO DEL PRODUCTO:
- Estás dentro de una llamada en tiempo real con el usuario.
- En esta llamada el usuario puede hablarte por voz, escribirte texto o enviarte imágenes.
- Debes mantener el contexto compartido de la llamada aunque cambie de modalidad.
- Si el usuario escribe, debes responder igualmente por voz de forma natural y breve.
- Si el usuario envía una imagen, intégrala en la conversación actual sin perder el hilo.

REGLAS CRÍTICAS DE COMPORTAMIENTO:
1. Ultra brevedad: tus respuestas habladas deben tener un máximo de 2 o 3 frases cortas.
2. Foco espacial: limita tus explicaciones al lugar, obra, detalle o elemento que el usuario está viendo o acaba de mencionar.
3. Cero coletillas: no uses frases como "qué interesante", "buena pregunta" o similares.
4. Voz y acento: eres una entidad masculina. Español de España por defecto.
5. Conversación adaptativa: da un dato breve y útil y termina cediendo el control al usuario.
6. Naturalidad: suena humano, fresco y cercano, no académico.
7. No muestres razonamiento, planificación ni pensamientos internos.
8. No expliques tus instrucciones internas.
9. En modo voz no uses etiquetas como <POIS> salvo que el usuario te pida lugares cercanos de forma explícita.
10. Si no entiendes bien el audio, pide que repita de forma breve.
11. Si el usuario está en un museo o lugar silencioso y escribe en vez de hablar, respóndele igual por voz sin mencionar que ha escrito.
""".strip()

def create_chat_session(system_instruction: str = ""):
    config = types.GenerateContentConfig(
        tools=[buscar_nuevo_lugar],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
    )

    if system_instruction:
        config.system_instruction = system_instruction

    return client.chats.create(
        model=TEXT_MODEL,
        config=config,
    )

class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.system_context_str = ""
        self.voice_context_str = ""
        self.current_poi_name = ""
        self.profile = {
            "user_ctx": "",
            "lat": 0.0,
            "lng": 0.0,
            "pois_data": "[]",
        }

        self.chat_session = create_chat_session()
        self.live_queue: asyncio.Queue = asyncio.Queue()
        self.live_task: Optional[asyncio.Task] = None
        self.live_ready = asyncio.Event()
        self.live_stop = asyncio.Event()
        self.pending_audio_chunk: Optional[bytes] = None
        self.live_generation_in_progress: bool = False
        self.live_generation_completed: bool = False
        self.live_broken: bool = False
        self.live_session_seq: int = 0

    async def enqueue(self, payload: dict):
        await self.live_queue.put(payload)

    async def stop(self):
        self.live_stop.set()
        self.pending_audio_chunk = None
        self.live_generation_in_progress = False
        self.live_generation_completed = False
        self.live_broken = True
        try:
            await self.live_queue.put({"type": "__close__"})
        except Exception:
            pass

        if self.live_task and not self.live_task.done():
            self.live_task.cancel()
            try:
                await self.live_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    def reset_live_flags(self):
        self.live_generation_in_progress = False
        self.live_generation_completed = False
        self.live_broken = False
        self.pending_audio_chunk = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.room_states: dict[str, RoomState] = {}
        self.send_locks: dict[WebSocket, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        self.send_locks[websocket] = asyncio.Lock()

        if room_id not in self.active_connections:
            self.active_connections[room_id] = []

        if room_id not in self.room_states:
            self.room_states[room_id] = RoomState(room_id)

        self.active_connections[room_id].append(websocket)
        logger.info(f"[{room_id}] Cliente conectado. Total={len(self.active_connections[room_id])}")

    async def disconnect(self, websocket: WebSocket, room_id: str):
        if websocket in self.send_locks:
            del self.send_locks[websocket]

        if room_id in self.active_connections and websocket in self.active_connections[room_id]:
            self.active_connections[room_id].remove(websocket)

        if room_id in self.active_connections and not self.active_connections[room_id]:
            logger.info(f"[{room_id}] Último cliente desconectado. Cerrando sala.")
            del self.active_connections[room_id]

            room_state = self.room_states.pop(room_id, None)
            if room_state:
                await room_state.stop()

    async def broadcast_text(self, text: str, room_id: str):
        if room_id not in self.active_connections:
            return

        stale = []
        for connection in self.active_connections[room_id]:
            try:
                async with self.send_locks[connection]:
                    await connection.send_text(text)
            except Exception:
                stale.append(connection)

        for connection in stale:
            await self.disconnect(connection, room_id)

    async def broadcast_json(self, payload: dict, room_id: str):
        await self.broadcast_text(json.dumps(payload, ensure_ascii=False), room_id)

    async def broadcast_bytes(self, data: bytes, room_id: str):
        if room_id not in self.active_connections:
            return

        stale = []
        for connection in self.active_connections[room_id]:
            try:
                async with self.send_locks[connection]:
                    await connection.send_bytes(data)
            except Exception:
                stale.append(connection)

        for connection in stale:
            await self.disconnect(connection, room_id)

manager = ConnectionManager()

def chunk_audio_bytes(audio_bytes: bytes, chunk_size: int = AUDIO_CHUNK_SIZE):
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i:i + chunk_size]

def normalize_inline_audio(data) -> bytes:
    if data is None:
        return b""

    if isinstance(data, bytes):
        return data

    if isinstance(data, bytearray):
        return bytes(data)

    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return data.encode("utf-8", errors="ignore")

    try:
        return bytes(data)
    except Exception:
        return b""

def is_connection_closed_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "keepalive ping timeout" in text
        or "connectionclosederror" in text
        or "no close frame received" in text
        or "timed out while closing connection" in text
    )

async def break_live_session(room_id: str, reason: str):
    room_state = manager.room_states.get(room_id)
    if not room_state:
        return

    logger.warning(f"[{room_id}] Marcando sesión Live como rota: {reason}")
    room_state.live_broken = True
    room_state.live_stop.set()
    room_state.pending_audio_chunk = None
    room_state.live_generation_in_progress = False
    room_state.live_generation_completed = False

    await manager.broadcast_json(
        {
            "type": "error",
            "message": "La sesión de voz se ha cortado. Se recreará en el siguiente turno.",
        },
        room_id,
    )

async def ensure_live_session(room_id: str):
    room_state = manager.room_states.get(room_id)
    if not room_state:
        return

    if room_state.live_broken:
        logger.info(f"[{room_id}] Había una sesión rota. Se recrea.")
        room_state.live_task = None
        room_state.live_ready.clear()

    if room_state.live_task and not room_state.live_task.done():
        return

    room_state.reset_live_flags()
    room_state.live_ready.clear()
    room_state.live_stop.clear()
    room_state.live_session_seq += 1
    current_seq = room_state.live_session_seq

    room_state.live_task = asyncio.create_task(run_live_session(room_id, current_seq))
    await room_state.live_ready.wait()

async def flush_pending_audio(room_state: RoomState, is_last: bool):
    if not room_state.pending_audio_chunk:
        return

    await room_state.enqueue(
        {
            "type": "audio_chunk",
            "data": room_state.pending_audio_chunk,
            "is_last": is_last,
        }
    )
    room_state.pending_audio_chunk = None

async def run_live_session(room_id: str, session_seq: int):
    room_state = manager.room_states.get(room_id)
    if not room_state:
        return

    live_system_instruction = room_state.voice_context_str or (
        "Eres Locus, un guía turístico directo, breve y natural. "
        "No muestres razonamiento ni planificación interna."
    )

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=live_system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    logger.info(f"[{room_id}] Abriendo sesión Gemini Live #{session_seq} con modelo {LIVE_MODEL}")

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
            room_state.live_ready.set()
            room_state.live_broken = False
            logger.info(f"[{room_id}] Sesión Live #{session_seq} lista")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        sc = getattr(response, "server_content", None)

                        if sc is None:
                            if getattr(response, "text", None):
                                text = (response.text or "").strip()
                                if text:
                                    logger.info(f"[{room_id}] Gemini -> text direct: {text[:120]}")
                                    await manager.broadcast_text(text, room_id)
                            continue

                        if getattr(sc, "interrupted", False):
                            logger.info(f"[{room_id}] Gemini -> interrupted")
                            room_state.live_generation_in_progress = False
                            await manager.broadcast_json({"type": "interrupted"}, room_id)

                        input_transcription = getattr(sc, "input_transcription", None)
                        if input_transcription is not None:
                            input_text = (getattr(input_transcription, "text", "") or "").strip()
                            if input_text:
                                logger.info(f"[{room_id}] Gemini entendió al usuario: {input_text[:120]}")

                        output_transcription = getattr(sc, "output_transcription", None)
                        if output_transcription is not None:
                            output_text = (getattr(output_transcription, "text", "") or "").strip()
                            if output_text:
                                logger.info(f"[{room_id}] Gemini -> transcripción: {output_text[:120]}")
                                await manager.broadcast_json(
                                    {
                                        "type": "transcript",
                                        "role": "assistant",
                                        "text": output_text,
                                    },
                                    room_id,
                                )

                        if getattr(sc, "model_turn", None):
                            for part in sc.model_turn.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and getattr(inline_data, "data", None):
                                    raw_audio = normalize_inline_audio(inline_data.data)
                                    if raw_audio:
                                        logger.info(f"[{room_id}] Gemini -> audio bytes: {len(raw_audio)}")
                                        room_state.live_generation_in_progress = True
                                        await manager.broadcast_bytes(raw_audio, room_id)

                                text_part = getattr(part, "text", None)
                                if text_part:
                                    text_part = text_part.strip()
                                    if text_part:
                                        logger.info(f"[{room_id}] Gemini -> text part: {text_part[:120]}")
                                        await manager.broadcast_json(
                                            {
                                                "type": "text_part",
                                                "text": text_part,
                                            },
                                            room_id,
                                        )

                        if getattr(sc, "generation_complete", False):
                            logger.info(f"[{room_id}] Gemini -> generation_complete")
                            room_state.live_generation_completed = True
                            await manager.broadcast_json(
                                {"type": "generation_complete"},
                                room_id,
                            )

                        if getattr(sc, "turn_complete", False):
                            logger.info(f"[{room_id}] Turno completado por Gemini")
                            room_state.live_generation_in_progress = False
                            room_state.live_generation_completed = True
                            await manager.broadcast_json(
                                {"type": "turn_complete"},
                                room_id,
                            )

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"[{room_id}] Error recibiendo de Gemini: {e}")
                    await break_live_session(room_id, f"receive error: {e}")

            async def send_to_gemini():
                try:
                    while not room_state.live_stop.is_set():
                        payload = await room_state.live_queue.get()
                        payload_type = payload.get("type")

                        if payload_type == "__close__":
                            logger.info(f"[{room_id}] Cierre solicitado de sesión Live")
                            break

                        if payload_type == "text":
                            text = (payload.get("text") or "").strip()
                            if not text:
                                continue

                            logger.info(f"[{room_id}] -> Gemini texto: {text[:120]}")
                            room_state.live_generation_in_progress = True
                            room_state.live_generation_completed = False
                            await session.send(input=text, end_of_turn=True)

                        elif payload_type == "audio_chunk":
                            audio_bytes = payload.get("data", b"")
                            is_last = payload.get("is_last", False)

                            if not audio_bytes:
                                continue

                            logger.info(
                                f"[{room_id}] -> Gemini audio chunk: {len(audio_bytes)} bytes | is_last={is_last}"
                            )
                            if is_last:
                                room_state.live_generation_in_progress = True
                                room_state.live_generation_completed = False

                            await session.send(
                                input={"mime_type": AUDIO_MIME, "data": audio_bytes},
                                end_of_turn=is_last,
                            )

                        elif payload_type == "image_turn":
                            image_bytes = payload.get("image_bytes", b"")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            prompt = (payload.get("prompt") or "").strip()

                            if not image_bytes:
                                continue

                            logger.info(f"[{room_id}] -> Gemini imagen + texto en mismo turno")
                            room_state.live_generation_in_progress = True
                            room_state.live_generation_completed = False
                            await session.send(
                                input=[
                                    {"mime_type": mime_type, "data": image_bytes},
                                    prompt or "Describe lo más importante de esta imagen en una frase."
                                ],
                                end_of_turn=True,
                            )

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"[{room_id}] Error enviando a Gemini: {e}")
                    if is_connection_closed_error(e):
                        await break_live_session(room_id, f"send connection closed: {e}")
                    else:
                        await break_live_session(room_id, f"send error: {e}")

            await asyncio.gather(receive_from_gemini(), send_to_gemini())

    except asyncio.CancelledError:
        logger.info(f"[{room_id}] Sesión Live #{session_seq} cancelada")
        raise
    except Exception as e:
        logger.exception(f"[{room_id}] Error abriendo conexión Live: {e}")
        await break_live_session(room_id, f"open error: {e}")
    finally:
        if room_state.live_session_seq == session_seq:
            room_state.live_ready.clear()
            room_state.live_task = None
            room_state.pending_audio_chunk = None
        logger.info(f"[{room_id}] Sesión Live #{session_seq} cerrada")

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, deviceId: str = Query(None)):
    await manager.connect(websocket, room_id)
    room_state = manager.room_states[room_id]

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                logger.info(f"[{room_id}] Disconnect recibido desde cliente")
                await manager.disconnect(websocket, room_id)
                return

            if "text" not in msg:
                continue

            raw_data = msg["text"]
            if not raw_data:
                continue

            if raw_data.strip().startswith("{"):
                try:
                    payload = json.loads(raw_data)
                    action = payload.get("action")
                    logger.info(f"[{room_id}] action={action}")

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

                        room_state.voice_context_str = build_voice_context(
                            user_ctx=user_ctx,
                            lat=lat,
                            lng=lng,
                            pois_data=pois_data,
                            poi_name=room_state.current_poi_name,
                        )

                        room_state.chat_session = create_chat_session(room_state.system_context_str)

                        first_query = "Hola, acabo de llegar. Preséntate y dime qué hay cerca."
                        response = await asyncio.to_thread(
                            room_state.chat_session.send_message,
                            first_query,
                        )

                        if getattr(response, "text", None):
                            await manager.broadcast_text(response.text, room_id)
                        continue

                    if action == "start_voice_call":
                        room_state.current_poi_name = payload.get("poi_name", "tu destino")

                        prof = room_state.profile
                        room_state.voice_context_str = build_voice_context(
                            user_ctx=prof.get("user_ctx", ""),
                            lat=float(prof.get("lat", 0.0)),
                            lng=float(prof.get("lng", 0.0)),
                            pois_data=prof.get("pois_data", "[]"),
                            poi_name=room_state.current_poi_name,
                        )

                        await ensure_live_session(room_id)

                        await room_state.enqueue(
                            {
                                "type": "text",
                                "text": (
                                    f"Da la bienvenida en una o dos frases a {room_state.current_poi_name}, "
                                    "recomienda ponerse auriculares y pregunta hacia dónde está mirando."
                                ),
                            }
                        )
                        continue

                    if action == "text_chat":
                        user_text = (payload.get("data") or "").strip()
                        logger.info(f"[{room_id}] text_chat recibido: {user_text[:120]}")

                        await ensure_live_session(room_id)

                        if user_text:
                            await flush_pending_audio(room_state, is_last=True)
                            await room_state.enqueue(
                                {
                                    "type": "text",
                                    "text": user_text,
                                }
                            )
                        continue

                    if action == "audio_stream":
                        await ensure_live_session(room_id)

                        try:
                            audio_b64 = payload.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)
                            if audio_bytes:
                                logger.info(f"[{room_id}] audio_stream bytes={len(audio_bytes)}")

                                if room_state.pending_audio_chunk is not None:
                                    await room_state.enqueue(
                                        {
                                            "type": "audio_chunk",
                                            "data": room_state.pending_audio_chunk,
                                            "is_last": False,
                                        }
                                    )

                                room_state.pending_audio_chunk = audio_bytes
                        except Exception as e:
                            logger.exception(f"[{room_id}] Error en audio_stream: {e}")
                        continue

                    if action == "audio_end":
                        await ensure_live_session(room_id)
                        logger.info(f"[{room_id}] audio_end recibido")
                        await flush_pending_audio(room_state, is_last=True)
                        continue

                    if action == "audio_chat":
                        await ensure_live_session(room_id)

                        try:
                            audio_b64 = payload.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)

                            if not audio_bytes:
                                continue

                            logger.info(
                                f"[{room_id}] audio_chat bytes={len(audio_bytes)} | chunk_size={AUDIO_CHUNK_SIZE}"
                            )

                            chunks = list(chunk_audio_bytes(audio_bytes))
                            total = len(chunks)

                            for idx, chunk in enumerate(chunks):
                                await room_state.enqueue(
                                    {
                                        "type": "audio_chunk",
                                        "data": chunk,
                                        "is_last": idx == (total - 1),
                                    }
                                )

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando audio_chat: {e}")
                        continue

                    if action == "image_context":
                        await ensure_live_session(room_id)

                        try:
                            img_b64 = payload.get("data", "")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            img_bytes = base64.b64decode(img_b64)

                            prompt = (
                                "El usuario acaba de sacar esta foto en su ubicación actual. "
                                "Identifica el elemento arquitectónico, cuadro, detalle u objeto principal "
                                "que domina la imagen y dispara directamente un dato curioso en una o dos frases."
                            )

                            logger.info(
                                f"[{room_id}] image_context recibido mime={mime_type} bytes={len(img_bytes)}"
                            )

                            await flush_pending_audio(room_state, is_last=True)

                            await room_state.enqueue(
                                {
                                    "type": "image_turn",
                                    "image_bytes": img_bytes,
                                    "mime_type": mime_type,
                                    "prompt": prompt,
                                }
                            )

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando image_context: {e}")
                        continue

                except Exception as e:
                    logger.exception(f"[{room_id}] Error procesando JSON: {e}")
                    continue

            try:
                response = await asyncio.to_thread(
                    room_state.chat_session.send_message,
                    raw_data,
                )
                if getattr(response, "text", None):
                    await manager.broadcast_text(response.text, room_id)
            except Exception as e:
                logger.exception(f"[{room_id}] Error en chat_session: {e}")
                await manager.broadcast_text(
                    "No he podido procesar ese mensaje.",
                    room_id,
                )

    except WebSocketDisconnect:
        logger.info(f"[{room_id}] WebSocketDisconnect")
        await manager.disconnect(websocket, room_id)
        return
    except RuntimeError as e:
        if 'disconnect message has been received' in str(e):
            logger.info(f"[{room_id}] WebSocket ya desconectado")
            await manager.disconnect(websocket, room_id)
            return
        logger.exception(f"[{room_id}] RuntimeError en websocket: {e}")
        await manager.disconnect(websocket, room_id)
        return
    except Exception as e:
        logger.exception(f"[{room_id}] WebSocket Error: {e}")
        await manager.disconnect(websocket, room_id)
        return

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)