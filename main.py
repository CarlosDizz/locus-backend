import os
import json
import urllib.parse
import urllib.request
import logging
import base64
import asyncio
import io
from pydub import AudioSegment
from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocusIA")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = "gemini-3.1-flash-lite-preview"
LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def buscar_nuevo_lugar(consulta: str, lat: float, lng: float) -> str:
    try:
        query = urllib.parse.quote(consulta)
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&location={lat},{lng}&radius=3000&key={MAPS_API_KEY}&language=es"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get('results', [])[:4]
            return json.dumps([{"name": r.get('name'), "lat": r['geometry']['location']['lat'], "lng": r['geometry']['location']['lng']} for r in results])
    except Exception as e:
        logger.error(f"Error en Maps: {e}")
        return "[]"

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.room_states: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
            self.room_states[room_id] = {
                "chat_session": client.chats.create(
                    model=TEXT_MODEL,
                    config=types.GenerateContentConfig(
                        tools=[buscar_nuevo_lugar],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
                    )
                ),
                "shadow_history": [],
                "system_context_str": "",
                "live_queue": asyncio.Queue(),
                "live_task": None
            }
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                if room_id in self.room_states:
                    if self.room_states[room_id]["live_task"]:
                        self.room_states[room_id]["live_task"].cancel()
                    del self.room_states[room_id]
                del self.active_connections[room_id]

    async def broadcast_text(self, text: str, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_text(text)
                except Exception: pass

    async def broadcast_bytes(self, data: bytes, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_bytes(data)
                except Exception: pass

manager = ConnectionManager()

async def run_live_session(room_id: str):
    room_state = manager.room_states.get(room_id)
    if not room_state: return

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')
            )
        )
    )

    try:
        logger.info(f"[{room_id}] Abriendo túnel Gemini Live...")
        async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as ephemeral_session:
            async def receive_from_gemini():
                async for response in ephemeral_session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                await manager.broadcast_text(part.text, room_id)
                            if part.inline_data:
                                # LOG: Gemini está respondiendo con audio
                                logger.info(f"[{room_id}] Gemini enviando audio ({len(part.inline_data.data)} bytes)")
                                await manager.broadcast_bytes(part.inline_data.data, room_id)

            async def send_to_gemini():
                while True:
                    payload = await room_state["live_queue"].get()
                    if payload is None: break

                    if "mime_type" in payload and "data" in payload:
                        logger.info(f"[{room_id}] Inyectando BINARIO en túnel Gemini...")
                        media_input = {"mime_type": payload["mime_type"], "data": payload["data"]}
                        is_audio = payload["mime_type"].startswith("audio/")
                        await ephemeral_session.send(input=media_input, end_of_turn=is_audio)
                    elif "text" in payload:
                        logger.info(f"[{room_id}] Inyectando TEXTO en túnel Gemini: {payload['text'][:50]}...")
                        await ephemeral_session.send(input=payload["text"], end_of_turn=True)

            await asyncio.gather(receive_from_gemini(), send_to_gemini())
    except Exception as e:
        logger.error(f"[{room_id}] Error en túnel: {e}")

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, deviceId: str = Query(None)):
    await manager.connect(websocket, room_id)
    room_state = manager.room_states[room_id]

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg:
                data = msg["text"]
                if data.strip().startswith('{'):
                    payload = json.loads(data)
                    action = payload.get("action")

                    if action == "setup_profile":
                        user_ctx = payload.get("context", "")
                        lat, lng = float(payload.get("lat", 0.0)), float(payload.get("lng", 0.0))
                        pois_data = buscar_nuevo_lugar("atracciones turisticas", lat, lng)
                        room_state["system_context_str"] = f"Eres Locus. Contexto: {user_ctx}. POIs: {pois_data}. Sé breve (2 frases) e interpela siempre."
                        first_query = f"{room_state['system_context_str']}\nPreséntate y dime qué hay cerca."
                        response = room_state["chat_session"].send_message(first_query)
                        await manager.broadcast_text(response.text, room_id)

                    elif action == "start_voice_call":
                        if not room_state["live_task"] or room_state["live_task"].done():
                            room_state["live_task"] = asyncio.create_task(run_live_session(room_id))
                        poi_name = payload.get("poi_name", "tu destino")
                        user_text = f"CONTEXTO: {room_state['system_context_str']}\nSaluda al usuario que llega a {poi_name}."
                        await room_state["live_queue"].put({"text": user_text})

                    elif action == "audio_chat":
                        try:
                            audio_bytes = base64.b64decode(payload.get("data"))
                            logger.info(f"[{room_id}] Recibido audio de cliente ({len(audio_bytes)} bytes)")

                            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                            pcm_data = audio_segment.raw_data

                            logger.info(f"[{room_id}] Audio convertido a PCM 16kHz ({len(pcm_data)} bytes)")
                            await room_state["live_queue"].put({"mime_type": "audio/pcm;rate=16000", "data": pcm_data})
                        except Exception as e:
                            logger.error(f"Error procesando audio: {e}")

                    elif action == "image_context":
                        img_b64 = payload.get("data")
                        logger.info(f"[{room_id}] Recibida imagen. Enviando a Gemini...")
                        safe_dict = {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                        await room_state["live_queue"].put({"image_dict": safe_dict})
                        await room_state["live_queue"].put({"text": "Identifica este detalle y dame un dato curioso."})

                else:
                    response = room_state["chat_session"].send_message(data)
                    await manager.broadcast_text(response.text, room_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        manager.disconnect(websocket, room_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)