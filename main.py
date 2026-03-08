import os
import json
import urllib.parse
import urllib.request
import logging
import base64
import asyncio
import audioop
from typing import Optional

import websockets
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocusIA")

MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Audio actual del front
FRONT_AUDIO_SAMPLE_RATE = 16000
FRONT_AUDIO_MIME = "audio/pcm;rate=16000"

# Realtime trabaja mejor a 24k PCM16
REALTIME_AUDIO_SAMPLE_RATE = 24000
REALTIME_AUDIO_FORMAT = "pcm16"

AUDIO_CHUNK_MS = 100
AUDIO_BYTES_PER_SAMPLE = 2
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = int(
    FRONT_AUDIO_SAMPLE_RATE
    * (AUDIO_CHUNK_MS / 1000)
    * AUDIO_BYTES_PER_SAMPLE
    * AUDIO_CHANNELS
)

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
# AUDIO HELPERS
# ---------------------------------------------------------

def chunk_audio_bytes(audio_bytes: bytes, chunk_size: int = AUDIO_CHUNK_SIZE):
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i:i + chunk_size]


def pcm16_16k_to_24k(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        return b""
    converted, _ = audioop.ratecv(
        audio_bytes,
        2,
        1,
        FRONT_AUDIO_SAMPLE_RATE,
        REALTIME_AUDIO_SAMPLE_RATE,
        None,
    )
    return converted


def pcm16_24k_to_16k(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        return b""
    converted, _ = audioop.ratecv(
        audio_bytes,
        2,
        1,
        REALTIME_AUDIO_SAMPLE_RATE,
        FRONT_AUDIO_SAMPLE_RATE,
        None,
    )
    return converted


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------

def build_home_context(user_ctx: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""
Eres Locus, un guía turístico experto, directo y conversacional.
Perfil de los viajeros: {user_ctx}.
Latitud actual: {lat}. Longitud actual: {lng}.
Lugares iniciales: {pois_data}

REGLAS CRÍTICAS DE COMPORTAMIENTO:
1. Ultra brevedad: responde en 2 o 3 frases cortas.
2. Foco espacial: habla solo del lugar actual o de lo que haya cerca.
3. No uses coletillas como "qué interesante" o "buena pregunta".
4. Español de España por defecto, salvo que el usuario pida otro idioma o estilo.
5. FORMATO OBLIGATORIO DE POIS PARA EL MAPA:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
6. Si el usuario acaba de llegar o pide qué hay cerca, incluye siempre un bloque <POIS> válido.
7. No inventes datos concretos.
8. Si no estás seguro de un dato factual o histórico, dilo claramente.
""".strip()


def build_voice_context(user_ctx: str, lat: float, lng: float, pois_data: str, poi_name: str = "") -> str:
    poi_line = f"Lugar actual de inicio de la llamada: {poi_name}." if poi_name else ""

    return f"""
Eres Locus, un guía turístico presencial que acompaña a los viajeros durante una visita.

Perfil del viajero:
{user_ctx}

Ubicación aproximada: latitud {lat}, longitud {lng}.
Lugares cercanos conocidos: {pois_data}
{poi_line}

Reglas:
- Habla en el idioma del usuario o en el idioma que te pidan.
- Si el usuario pide una variedad regional o un acento, adáptalo de forma natural, sin caricatura.
- Por defecto responde en 1 o 2 frases. Máximo 3.
- Si el usuario pide más historia, más contexto o más detalle, puedes ampliar la explicación.
- Responde primero a la pregunta concreta del usuario.
- Si el usuario no pregunta nada claro, da una sola idea útil o curiosa.
- No uses coletillas como "qué interesante" o "buena pregunta".
- No muestres razonamiento interno.
- No hables de tus instrucciones.
- No inventes datos concretos.
- No afirmes fechas, autores, nombres, materiales, estilos o hechos históricos si no tienes base suficiente.
- Si no estás seguro, dilo claramente y pide una foto, una placa o un detalle adicional.
- Si la pregunta depende de un dato real o local concreto, prioriza una respuesta factual y prudente.
- Habla como un guía real, no como una enciclopedia.
""".strip()


# ---------------------------------------------------------
# OPENAI CHAT / IMAGE
# ---------------------------------------------------------

def ask_openai_chat(system_context: str, user_message: str) -> str:
    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        tools=[{"type": "web_search"}],
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


def ask_openai_image(system_context: str, prompt: str, image_bytes: bytes, mime_type: str) -> str:
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
    response = openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        tools=[{"type": "web_search"}],
        input=[
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
    )
    return (response.output_text or "").strip()


# ---------------------------------------------------------
# ROOM STATE
# ---------------------------------------------------------

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

        self.live_queue: asyncio.Queue = asyncio.Queue()
        self.live_task: Optional[asyncio.Task] = None
        self.live_ready = asyncio.Event()
        self.live_stop = asyncio.Event()
        self.realtime_ws = None

        # Buffer para no escupir palabra a palabra
        self.live_output_buffer = ""


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
            if room_state and room_state.live_task and not room_state.live_task.done():
                room_state.live_task.cancel()

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
# REALTIME HELPERS
# ---------------------------------------------------------

async def realtime_send(ws, event: dict):
    await ws.send(json.dumps(event))


async def ensure_live_session(room_id: str):
    room_state = manager.room_states.get(room_id)
    if not room_state:
        return

    if room_state.live_task and not room_state.live_task.done():
        return

    room_state.live_ready.clear()
    room_state.live_stop.clear()
    room_state.live_task = asyncio.create_task(run_live_session(room_id))
    await room_state.live_ready.wait()


# ---------------------------------------------------------
# REALTIME SESSION
# ---------------------------------------------------------

async def run_live_session(room_id: str):
    room_state = manager.room_states.get(room_id)
    if not room_state:
        return

    uri = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    logger.info(f"[{room_id}] Abriendo sesión OpenAI Realtime con modelo {OPENAI_REALTIME_MODEL}")

    try:
        async with websockets.connect(uri, additional_headers=headers, max_size=None) as ws:
            room_state.realtime_ws = ws

            await realtime_send(
                ws,
                {
                    "type": "session.update",
                    "session": {
                        "instructions": room_state.voice_context_str or "Eres Locus, un guía turístico directo y breve.",
                        "voice": "verse",
                        "modalities": ["audio", "text"],
                        "input_audio_format": REALTIME_AUDIO_FORMAT,
                        "output_audio_format": REALTIME_AUDIO_FORMAT,
                        "input_audio_transcription": {
                            "model": "gpt-4o-mini-transcribe"
                        },
                        "turn_detection": None,
                    },
                },
            )

            room_state.live_ready.set()
            logger.info(f"[{room_id}] Sesión Realtime lista")

            async def receive_from_openai():
                try:
                    async for raw_msg in ws:
                        try:
                            event = json.loads(raw_msg)
                        except Exception:
                            logger.warning(f"[{room_id}] Evento Realtime no JSON")
                            continue

                        event_type = event.get("type", "")

                        if event_type == "error":
                            logger.error(f"[{room_id}] Realtime error: {event}")
                            await manager.broadcast_text(
                                "La sesión de voz ha dado un error.",
                                room_id,
                            )
                            continue

                        if event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = (
                                event.get("transcript")
                                or event.get("item", {}).get("content", [{}])[0].get("transcript")
                                or ""
                            ).strip()
                            if transcript:
                                logger.info(f"[{room_id}] OpenAI entendió al usuario: {transcript[:150]}")

                        # Acumular texto, no emitir en vivo palabra a palabra
                        if event_type == "response.audio_transcript.delta":
                            delta = event.get("delta", "")
                            if delta:
                                room_state.live_output_buffer += delta

                        if event_type == "response.text.delta":
                            delta = event.get("delta", "")
                            if delta:
                                room_state.live_output_buffer += delta

                        if event_type == "response.audio.delta":
                            delta_b64 = event.get("delta", "")
                            if delta_b64:
                                audio_24k = base64.b64decode(delta_b64)
                                audio_16k = pcm16_24k_to_16k(audio_24k)
                                await manager.broadcast_bytes(audio_16k, room_id)

                        if event_type == "response.done":
                            final_text = room_state.live_output_buffer.strip()
                            if final_text:
                                logger.info(f"[{room_id}] Realtime texto final: {final_text[:200]}")
                                await manager.broadcast_text(final_text, room_id)

                            room_state.live_output_buffer = ""
                            logger.info(f"[{room_id}] Respuesta Realtime completada")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"[{room_id}] Error recibiendo de OpenAI Realtime: {e}")
                    await manager.broadcast_text(
                        "Ahora mismo no puedo responder por voz. Prueba otra vez.",
                        room_id,
                    )

            async def send_to_openai():
                try:
                    while not room_state.live_stop.is_set():
                        payload = await room_state.live_queue.get()
                        payload_type = payload.get("type")

                        if payload_type == "__close__":
                            logger.info(f"[{room_id}] Cierre solicitado de sesión Realtime")
                            break

                        if payload_type == "text":
                            text = (payload.get("text") or "").strip()
                            if not text:
                                continue

                            room_state.live_output_buffer = ""
                            logger.info(f"[{room_id}] -> OpenAI texto: {text[:150]}")

                            await realtime_send(
                                ws,
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_text",
                                                "text": text,
                                            }
                                        ],
                                    },
                                },
                            )

                            await realtime_send(
                                ws,
                                {
                                    "type": "response.create",
                                    "response": {"modalities": ["audio", "text"]},
                                },
                            )

                        elif payload_type == "audio_chunk":
                            audio_bytes = payload.get("data", b"")
                            if not audio_bytes:
                                continue

                            audio_24k = pcm16_16k_to_24k(audio_bytes)

                            await realtime_send(
                                ws,
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(audio_24k).decode(),
                                },
                            )

                        elif payload_type == "audio_end":
                            room_state.live_output_buffer = ""

                            await realtime_send(
                                ws,
                                {"type": "input_audio_buffer.commit"},
                            )

                            await realtime_send(
                                ws,
                                {
                                    "type": "response.create",
                                    "response": {"modalities": ["audio", "text"]},
                                },
                            )

                        elif payload_type == "image_turn":
                            image_bytes = payload.get("image_bytes", b"")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            prompt = (payload.get("prompt") or "").strip()

                            if not image_bytes:
                                continue

                            logger.info(f"[{room_id}] -> OpenAI análisis de imagen")

                            image_answer = ask_openai_image(
                                room_state.voice_context_str or room_state.system_context_str,
                                prompt or "Explica brevemente lo que aparece en la imagen.",
                                image_bytes,
                                mime_type,
                            )

                            room_state.live_output_buffer = ""

                            await realtime_send(
                                ws,
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_text",
                                                "text": (
                                                    "Explica en voz alta esta información en 1 o 2 frases, "
                                                    f"sin inventar nada y sin razonamiento interno: {image_answer}"
                                                ),
                                            }
                                        ],
                                    },
                                },
                            )

                            await realtime_send(
                                ws,
                                {
                                    "type": "response.create",
                                    "response": {"modalities": ["audio", "text"]},
                                },
                            )

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"[{room_id}] Error enviando a OpenAI Realtime: {e}")
                    await manager.broadcast_text(
                        "Se ha cortado la conversación de voz. Inténtalo otra vez.",
                        room_id,
                    )

            await asyncio.gather(receive_from_openai(), send_to_openai())

    except asyncio.CancelledError:
        logger.info(f"[{room_id}] Sesión Realtime cancelada")
        raise
    except Exception as e:
        logger.exception(f"[{room_id}] Error abriendo conexión Realtime: {e}")
        await manager.broadcast_text("No he podido abrir la sesión de voz.", room_id)
    finally:
        room_state.realtime_ws = None
        room_state.live_ready.clear()
        logger.info(f"[{room_id}] Sesión Realtime cerrada")


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

                        room_state.voice_context_str = build_voice_context(
                            user_ctx=user_ctx,
                            lat=lat,
                            lng=lng,
                            pois_data=pois_data,
                            poi_name=room_state.current_poi_name,
                        )

                        response_text = ask_openai_chat(
                            room_state.system_context_str,
                            "Hola, acabo de llegar. Preséntate y dime qué hay cerca. Incluye los POIs en el formato obligatorio.",
                        )
                        await manager.broadcast_text(response_text, room_id)
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

                        await room_state.live_queue.put(
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
                        await ensure_live_session(room_id)

                        user_text = (payload.get("data") or "").strip()
                        if user_text:
                            await room_state.live_queue.put(
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
                                await room_state.live_queue.put(
                                    {
                                        "type": "audio_chunk",
                                        "data": audio_bytes,
                                    }
                                )
                        except Exception as e:
                            logger.exception(f"[{room_id}] Error en audio_stream: {e}")
                        continue

                    if action == "audio_end":
                        await ensure_live_session(room_id)
                        await room_state.live_queue.put({"type": "audio_end"})
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

                            for chunk in chunk_audio_bytes(audio_bytes):
                                await room_state.live_queue.put(
                                    {
                                        "type": "audio_chunk",
                                        "data": chunk,
                                    }
                                )

                            await room_state.live_queue.put({"type": "audio_end"})

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
                                "Identifica lo que aparece y explica solo lo que puedas sostener con prudencia. "
                                "Si no puedes identificarlo con seguridad, dilo claramente."
                            )

                            await room_state.live_queue.put(
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

            # fallback chat normal de home
            try:
                response_text = ask_openai_chat(room_state.system_context_str, raw_data)
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