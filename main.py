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
    "gemini-2.5-flash-native-audio-preview-12-2025"
)

AUDIO_MIME = "audio/pcm;rate=16000;channels=1"
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


class RoomState:

    def __init__(self, room_id: str):

        self.room_id = room_id

        self.chat_session = client.chats.create(
            model=TEXT_MODEL,
            config=types.GenerateContentConfig(
                tools=[buscar_nuevo_lugar],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=False
                ),
            ),
        )

        self.system_context_str = ""

        self.live_queue: asyncio.Queue = asyncio.Queue()

        self.live_task: Optional[asyncio.Task] = None

        self.live_ready = asyncio.Event()

        self.live_stop = asyncio.Event()

    async def enqueue(self, payload: dict):

        await self.live_queue.put(payload)

    async def stop(self):

        self.live_stop.set()

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

        logger.info(
            f"[{room_id}] Cliente conectado. Total={len(self.active_connections[room_id])}"
        )

    async def disconnect(self, websocket: WebSocket, room_id: str):

        if room_id in self.active_connections:

            if websocket in self.active_connections[room_id]:

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

        for connection in self.active_connections[room_id]:

            try:
                await connection.send_text(text)
            except Exception:
                pass

    async def broadcast_bytes(self, data: bytes, room_id: str):

        if room_id not in self.active_connections:
            return

        for connection in self.active_connections[room_id]:

            try:
                await connection.send_bytes(data)
            except Exception:
                pass


manager = ConnectionManager()


def chunk_audio_bytes(audio_bytes: bytes, chunk_size: int = AUDIO_CHUNK_SIZE):

    for i in range(0, len(audio_bytes), chunk_size):

        yield audio_bytes[i : i + chunk_size]


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


async def run_live_session(room_id: str):

    room_state = manager.room_states.get(room_id)

    if not room_state:
        return

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    logger.info(f"[{room_id}] Abriendo sesión Gemini Live con modelo {LIVE_MODEL}")

    try:

        async with client.aio.live.connect(
            model=LIVE_MODEL, config=live_config
        ) as session:

            room_state.live_ready.set()

            logger.info(f"[{room_id}] Sesión Live lista")

            async def receive_from_gemini():

                try:

                    async for response in session.receive():

                        sc = getattr(response, "server_content", None)

                        if sc is None:
                            continue

                        if getattr(sc, "output_transcription", None):

                            text = sc.output_transcription.text

                            if text:

                                logger.info(
                                    f"[{room_id}] Gemini -> texto: {text[:120]}"
                                )

                                await manager.broadcast_text(text, room_id)

                        if sc.model_turn:

                            for part in sc.model_turn.parts:

                                if getattr(part, "inline_data", None):

                                    data = part.inline_data.data

                                    if data:

                                        await manager.broadcast_bytes(data, room_id)

                        if sc.turn_complete:

                            logger.info(f"[{room_id}] Turno completado por Gemini")

                except asyncio.CancelledError:
                    raise

                except Exception as e:

                    logger.exception(f"[{room_id}] Error recibiendo de Gemini: {e}")

            async def send_to_gemini():

                try:

                    while not room_state.live_stop.is_set():

                        payload = await room_state.live_queue.get()

                        payload_type = payload.get("type")

                        if payload_type == "__close__":
                            break

                        if payload_type == "text":

                            text = payload.get("text", "").strip()

                            if not text:
                                continue

                            await session.send(input=text, end_of_turn=True)

                        elif payload_type == "audio_chunk":

                            audio_bytes = payload.get("data", b"")

                            is_last = payload.get("is_last", False)

                            if not audio_bytes:
                                continue

                            await session.send(
                                input={
                                    "mime_type": AUDIO_MIME,
                                    "data": audio_bytes,
                                },
                                end_of_turn=is_last,
                            )

                        elif payload_type == "image_turn":

                            image_bytes = payload.get("image_bytes", b"")

                            mime_type = payload.get("mime_type", "image/jpeg")

                            prompt = payload.get("prompt", "").strip()

                            if not image_bytes:
                                continue

                            await session.send(
                                input=[
                                    {"mime_type": mime_type, "data": image_bytes},
                                    prompt
                                    or "Describe lo más importante de esta imagen.",
                                ],
                                end_of_turn=True,
                            )

                except asyncio.CancelledError:
                    raise

                except Exception as e:

                    logger.exception(f"[{room_id}] Error enviando a Gemini: {e}")

            await asyncio.gather(receive_from_gemini(), send_to_gemini())

    except Exception as e:

        logger.exception(f"[{room_id}] Error abriendo conexión Live: {e}")

    finally:

        room_state.live_ready.clear()

        logger.info(f"[{room_id}] Sesión Live cerrada")


@app.websocket("/ws/{room_id}")

async def websocket_endpoint(
    websocket: WebSocket, room_id: str, deviceId: str = Query(None)
):

    await manager.connect(websocket, room_id)

    room_state = manager.room_states[room_id]

    try:

        while True:

            msg = await websocket.receive()

            if "text" not in msg:
                continue

            raw_data = msg["text"]

            if raw_data.strip().startswith("{"):

                payload = json.loads(raw_data)

                action = payload.get("action")

                if action == "start_voice_call":

                    await ensure_live_session(room_id)

                    await room_state.enqueue(
                        {
                            "type": "text",
                            "text": "Hola, soy tu guía Locus. ¿Qué estás viendo ahora mismo?",
                        }
                    )

                elif action == "text_chat":

                    await ensure_live_session(room_id)

                    await room_state.enqueue(
                        {"type": "text", "text": payload.get("data")}
                    )

                elif action == "audio_chat":

                    await ensure_live_session(room_id)

                    audio_b64 = payload.get("data")

                    audio_bytes = base64.b64decode(audio_b64)

                    logger.info(
                        f"[{room_id}] audio_chat bytes={len(audio_bytes)} | chunk_size={AUDIO_CHUNK_SIZE}"
                    )

                    chunks = list(chunk_audio_bytes(audio_bytes))

                    for i, chunk in enumerate(chunks):

                        await room_state.enqueue(
                            {
                                "type": "audio_chunk",
                                "data": chunk,
                                "is_last": i == len(chunks) - 1,
                            }
                        )

                elif action == "image_context":

                    await ensure_live_session(room_id)

                    img_b64 = payload.get("data")

                    mime_type = payload.get("mime_type", "image/jpeg")

                    img_bytes = base64.b64decode(img_b64)

                    await room_state.enqueue(
                        {
                            "type": "image_turn",
                            "image_bytes": img_bytes,
                            "mime_type": mime_type,
                            "prompt": "El usuario acaba de sacar esta foto. Dile un dato curioso breve.",
                        }
                    )

    except WebSocketDisconnect:

        await manager.disconnect(websocket, room_id)


if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)