import os
import json
import urllib.parse
import urllib.request
import logging
import base64
import asyncio
import struct
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
                try:
                    async for response in ephemeral_session.receive():
                        sc = response.server_content
                        if sc is not None:
                            if sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if part.text:
                                        logger.info(f"[{room_id}] IA responde con TEXTO.")
                                        await manager.broadcast_text(part.text, room_id)
                                    if part.inline_data:
                                        await manager.broadcast_bytes(part.inline_data.data, room_id)
                            if sc.turn_complete:
                                logger.info(f"[{room_id}] <-- IA terminó de hablar/procesar su turno.")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"[{room_id}] Error en recepción: {e}")

            async def send_to_gemini():
                try:
                    while True:
                        payload = await room_state["live_queue"].get()
                        if payload is None: break

                        if "image_dict" in payload:
                            await ephemeral_session.send(input=payload["image_dict"], end_of_turn=False)

                        elif "mime_type" in payload and "data" in payload:
                            media_input = {"mime_type": payload["mime_type"], "data": payload["data"]}
                            is_audio = payload["mime_type"].startswith("audio/")

                            logger.info(f"[{room_id}] Inyectando {payload['mime_type']} en túnel...")
                            await ephemeral_session.send(input=media_input, end_of_turn=is_audio)

                            if is_audio:
                                logger.info(f"[{room_id}] Turno cerrado. Esperando respuesta de la IA...")

                        elif "text" in payload:
                            logger.info(f"[{room_id}] Inyectando TEXTO en túnel: {payload['text'][:50]}...")
                            await ephemeral_session.send(input=payload["text"], end_of_turn=True)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"[{room_id}] Error en envío: {e}")

            await asyncio.gather(receive_from_gemini(), send_to_gemini())
    except Exception as e:
        logger.error(f"[{room_id}] Error en conexión Live: {e}")

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
                    try:
                        payload = json.loads(data)
                        action = payload.get("action")

                        if action == "setup_profile":
                            user_ctx = payload.get("context", "")
                            lat, lng = float(payload.get("lat", 0.0)), float(payload.get("lng", 0.0))
                            pois_data = buscar_nuevo_lugar("atracciones turisticas", lat, lng)

                            room_state["system_context_str"] = f"""
Eres Locus, un guía turístico experto, directo y conversacional.
Perfil de los viajeros: {user_ctx}.
Latitud actual: {lat}. Longitud actual: {lng}.
Lugares iniciales: {pois_data}

REGLAS CRÍTICAS DE COMPORTAMIENTO:
1. Ultra Brevedad: Tus respuestas habladas deben tener un máximo de 2 o 3 frases cortas. Eres un acompañante, no una enciclopedia.
2. Foco Espacial: Limita tus explicaciones ESTRICTAMENTE al lugar o monumento que el usuario está visitando en este momento. Si te preguntan algo fuera de contexto, devuélvelos al lugar actual.
3. Cero Coletillas: Nunca uses frases como '¡Qué interesante!' o 'Buena pregunta'. Ve directo a la información técnica o histórica.
4. Voz y Acento: Eres una entidad masculina. Adapta tu acento y vocabulario de forma natural al origen de los viajeros (Español de España por defecto).
5. FORMATO OBLIGATORIO DE POIS PARA EL MAPA:
<POIS>[{{"name": "Nombre", "lat": 0.0, "lng": 0.0, "description": "Descripción"}}]</POIS>
6. REGLA DE CONVERSACIÓN ADAPTATIVA (PÍLDORAS):
Nunca des explicaciones largas de golpe. Tu estructura obligatoria es:
- Da un solo dato fascinante o la idea principal en 1 o 2 frases.
- Termina SIEMPRE interpelando al usuario para cederle el control.
"""
                            first_query = f"{room_state['system_context_str']}\nUsuario: Hola, acabo de llegar. Preséntate y dime qué hay cerca."
                            response = room_state["chat_session"].send_message(first_query)
                            await manager.broadcast_text(response.text, room_id)
                            continue

                        if action == "start_voice_call":
                            if not room_state.get("live_task") or room_state["live_task"].done():
                                room_state["live_task"] = asyncio.create_task(run_live_session(room_id))
                            poi_name = payload.get("poi_name", "tu destino")
                            user_text = f"INSTRUCCIONES DE COMPORTAMIENTO:\n{room_state['system_context_str']}\n\nSITUACIÓN ACTUAL:\nEl usuario acaba de iniciar la ruta en {poi_name}. En un máximo de 2 frases: dale la bienvenida a este lugar exacto, recomiéndale ponerse los auriculares para aislarse del ruido, y pregúntale hacia dónde está mirando."
                            await room_state["live_queue"].put({"text": user_text})
                            continue

                        if action == "text_chat":
                            user_text = payload.get("data")
                            await room_state["live_queue"].put({"text": user_text})
                            continue

                        if action == "audio_chat":
                            try:
                                audio_bytes = base64.b64decode(payload.get("data"))

                                # OSCILOSCOPIO MATEMÁTICO: Calcular volumen máximo
                                valid_length = (len(audio_bytes) // 2) * 2
                                samples = struct.unpack(f"<{valid_length // 2}h", audio_bytes[:valid_length])
                                max_amp = max(abs(s) for s in samples) if samples else 0
                                is_silent = "SÍ (Microfono bloqueado/vacío)" if max_amp < 150 else "NO (Hay sonido)"

                                logger.info(f"[{room_id}] PCM Recibido: {len(audio_bytes)} bytes | Amplitud Max: {max_amp}/32768 | Silencio: {is_silent}")

                                await room_state["live_queue"].put({
                                    "mime_type": "audio/pcm;rate=16000",
                                    "data": audio_bytes
                                })
                            except Exception as e:
                                logger.error(f"Error procesando audio: {e}")
                            continue

                        if action == "image_context":
                            img_b64 = payload.get("data")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            img_bytes = base64.b64decode(img_b64)
                            user_text = "El usuario acaba de sacar esta foto en su ubicación actual. Identifica el elemento arquitectónico, cuadro o detalle que domina la imagen y dispara directamente un dato curioso en una sola frase."
                            logger.info(f"[{room_id}] Recibida imagen. Enviando a Gemini...")
                            await room_state["live_queue"].put({"mime_type": mime_type, "data": img_bytes})
                            await room_state["live_queue"].put({"text": user_text})
                            continue

                    except Exception as e:
                        logger.error(f"Error procesando JSON: {e}")
                        continue

                response = room_state["chat_session"].send_message(data)
                await manager.broadcast_text(response.text, room_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        manager.disconnect(websocket, room_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)