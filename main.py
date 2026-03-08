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

FRONT_AUDIO_SAMPLE_RATE = 16000
REALTIME_AUDIO_SAMPLE_RATE = 24000

AUDIO_CHUNK_MS = 100
AUDIO_BYTES_PER_SAMPLE = 2

AUDIO_CHUNK_SIZE = int(
    FRONT_AUDIO_SAMPLE_RATE
    * (AUDIO_CHUNK_MS / 1000)
    * AUDIO_BYTES_PER_SAMPLE
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------
# MAPS
# -----------------------------------------------------

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

        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = data.get("results", [])[:4]

        payload = []
        for r in results:
            loc = r["geometry"]["location"]
            payload.append(
                {
                    "name": r.get("name"),
                    "lat": loc["lat"],
                    "lng": loc["lng"],
                }
            )

        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.error(e)
        return "[]"


# -----------------------------------------------------
# AUDIO
# -----------------------------------------------------

def pcm16_16k_to_24k(audio_bytes: bytes) -> bytes:
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
    converted, _ = audioop.ratecv(
        audio_bytes,
        2,
        1,
        REALTIME_AUDIO_SAMPLE_RATE,
        FRONT_AUDIO_SAMPLE_RATE,
        None,
    )
    return converted


def chunk_audio_bytes(audio_bytes: bytes):
    for i in range(0, len(audio_bytes), AUDIO_CHUNK_SIZE):
        yield audio_bytes[i:i + AUDIO_CHUNK_SIZE]


# -----------------------------------------------------
# PROMPTS
# -----------------------------------------------------

def build_voice_context(user_ctx, lat, lng, pois, poi_name=""):

    return f"""
Eres Locus, un guía turístico presencial que acompaña a los viajeros durante una visita.

Perfil del viajero:
{user_ctx}

Ubicación aproximada: {lat}, {lng}

Lugares cercanos:
{pois}

Lugar actual:
{poi_name}

Reglas:

- Habla en el idioma del usuario.
- Si el usuario pide un acento o variante regional, adáptalo de forma natural.

- Por defecto responde en 1 o 2 frases.
- Si el usuario pide más historia o contexto puedes ampliar.

- Responde primero a la pregunta concreta.

- Si el usuario describe algo visual, intenta identificarlo.

- No inventes datos históricos.
- No inventes fechas ni nombres.
- Si no estás seguro dilo claramente.

- Si necesitas más contexto pide una foto o una placa.

- Habla como un guía real acompañando la visita.

- Evita coletillas como "qué interesante" o "buena pregunta".
""".strip()


# -----------------------------------------------------
# OPENAI CHAT
# -----------------------------------------------------

def ask_openai_chat(system_context, user_message):

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

    return response.output_text.strip()


# -----------------------------------------------------
# ROOM STATE
# -----------------------------------------------------

class RoomState:

    def __init__(self, room_id):

        self.room_id = room_id
        self.system_context_str = ""
        self.voice_context_str = ""

        self.live_queue = asyncio.Queue()
        self.live_task: Optional[asyncio.Task] = None
        self.live_ready = asyncio.Event()
        self.live_stop = asyncio.Event()

        self.realtime_ws = None

        self.output_buffer = ""


# -----------------------------------------------------
# CONNECTION MANAGER
# -----------------------------------------------------

class ConnectionManager:

    def __init__(self):
        self.active_connections = {}
        self.room_states = {}

    async def connect(self, websocket: WebSocket, room_id: str):

        await websocket.accept()

        if room_id not in self.active_connections:
            self.active_connections[room_id] = []

        if room_id not in self.room_states:
            self.room_states[room_id] = RoomState(room_id)

        self.active_connections[room_id].append(websocket)

    async def broadcast_text(self, text: str, room_id: str):

        for connection in self.active_connections.get(room_id, []):
            await connection.send_text(text)

    async def broadcast_bytes(self, data: bytes, room_id: str):

        for connection in self.active_connections.get(room_id, []):
            await connection.send_bytes(data)


manager = ConnectionManager()


# -----------------------------------------------------
# REALTIME SESSION
# -----------------------------------------------------

async def ensure_live_session(room_id):

    room = manager.room_states[room_id]

    if room.live_task and not room.live_task.done():
        return

    room.live_task = asyncio.create_task(run_live_session(room_id))

    await room.live_ready.wait()


async def realtime_send(ws, event):
    await ws.send(json.dumps(event))


async def run_live_session(room_id):

    room = manager.room_states[room_id]

    uri = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(uri, additional_headers=headers) as ws:

        room.realtime_ws = ws

        await realtime_send(
            ws,
            {
                "type": "session.update",
                "session": {
                    "instructions": room.voice_context_str,
                    "voice": "verse",
                    "modalities": ["audio", "text"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                },
            },
        )

        room.live_ready.set()

        async def receive_loop():

            async for msg in ws:

                event = json.loads(msg)

                t = event.get("type")

                if t == "response.audio.delta":

                    audio = base64.b64decode(event["delta"])

                    audio = pcm16_24k_to_16k(audio)

                    await manager.broadcast_bytes(audio, room_id)

                elif t == "response.text.delta":

                    room.output_buffer += event.get("delta", "")

                elif t == "response.done":

                    text = room.output_buffer.strip()

                    if text:
                        await manager.broadcast_text(text, room_id)

                    room.output_buffer = ""

        async def send_loop():

            while True:

                payload = await room.live_queue.get()

                t = payload["type"]

                if t == "text":

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
                                        "text": payload["text"],
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

                elif t == "audio_chunk":

                    audio24 = pcm16_16k_to_24k(payload["data"])

                    await realtime_send(
                        ws,
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio24).decode(),
                        },
                    )

                elif t == "audio_end":

                    await realtime_send(
                        ws,
                        {"type": "input_audio_buffer.commit"},
                    )

                    await realtime_send(
                        ws,
                        {"type": "response.create"},
                    )

        await asyncio.gather(receive_loop(), send_loop())


# -----------------------------------------------------
# WEBSOCKET
# -----------------------------------------------------

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, deviceId: str = Query(None)):

    await manager.connect(websocket, room_id)

    room = manager.room_states[room_id]

    try:

        while True:

            msg = await websocket.receive_text()

            payload = json.loads(msg)

            action = payload.get("action")

            if action == "setup_profile":

                user_ctx = payload.get("context", "")
                lat = float(payload.get("lat", 0))
                lng = float(payload.get("lng", 0))

                pois = buscar_nuevo_lugar("atracciones turisticas", lat, lng)

                room.system_context_str = build_voice_context(
                    user_ctx,
                    lat,
                    lng,
                    pois,
                )

                text = ask_openai_chat(
                    room.system_context_str,
                    "Hola, preséntate como guía y dime qué hay cerca.",
                )

                await manager.broadcast_text(text, room_id)

            elif action == "start_voice_call":

                await ensure_live_session(room_id)

            elif action == "text_chat":

                await room.live_queue.put(
                    {
                        "type": "text",
                        "text": payload["data"],
                    }
                )

            elif action == "audio_chat":

                audio = base64.b64decode(payload["data"])

                for chunk in chunk_audio_bytes(audio):

                    await room.live_queue.put(
                        {
                            "type": "audio_chunk",
                            "data": chunk,
                        }
                    )

                await room.live_queue.put({"type": "audio_end"})

    except WebSocketDisconnect:

        pass


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )