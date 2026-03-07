import os
import json
import urllib.parse
import urllib.request
import logging
import base64
import asyncio
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

TEXT_MODEL = "gemini-2.5-flash"
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
                "system_context_str": ""
            }
            logger.info(f"Sala creada: {room_id}")
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
                if room_id in self.room_states:
                    del self.room_states[room_id]
                logger.info(f"Sala destruida y memoria purgada: {room_id}")

    async def broadcast_text(self, text: str, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_text(text)
                except Exception:
                    pass

    async def broadcast_bytes(self, data: bytes, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_bytes(data)
                except Exception:
                    pass

manager = ConnectionManager()

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, deviceId: str = Query(None)):
    await manager.connect(websocket, room_id)

    room_state = manager.room_states[room_id]
    chat_session = room_state["chat_session"]

    async def process_voice_turn(user_text: str, image_bytes: bytes = None, mime_type: str = None):
        base_history = []
        try:
            for m in chat_session.get_history():
                if m.parts and m.parts[0].text:
                    base_history.append(f"{m.role}: {m.parts[0].text}")
        except Exception:
            pass

        historial_base = "\n".join(base_history[-10:])
        historial_llamada = "\n".join(room_state["shadow_history"][-10:])

        prompt = f"""INSTRUCCIONES DE SISTEMA:
{room_state['system_context_str']}

REGLA CRÍTICA ABSOLUTA: Eres Locus. Habla SIEMPRE de forma directa con el usuario. NO uses asteriscos. NO pienses en voz alta. NO generes monólogos internos. Di únicamente el texto final que el usuario va a escuchar de tu voz.

HISTORIAL DEL CHAT ESCRITO:
{historial_base}

HISTORIAL DE LA LLAMADA DE VOZ:
{historial_llamada}

NUEVO MENSAJE DEL USUARIO:
{user_text}

LOCUS:"""

        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')
                )
            )
        )

        logger.info("Abriendo túnel efímero...")
        model_text = ""
        try:
            async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as ephemeral_session:
                if image_bytes and mime_type:
                    await ephemeral_session.send(input={"mime_type": mime_type, "data": image_bytes})

                await ephemeral_session.send(input=prompt, end_of_turn=True)

                async for response in ephemeral_session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                model_text += part.text
                                await manager.broadcast_text(part.text, room_id)
                            if part.inline_data:
                                await manager.broadcast_bytes(part.inline_data.data, room_id)
            logger.info("Túnel efímero cerrado con éxito tras respuesta.")
        except Exception as e:
            logger.error(f"Error en sesión efímera: {e}")

        if model_text:
            room_state["shadow_history"].append(f"Usuario: {user_text}")
            room_state["shadow_history"].append(f"Locus: {model_text.strip()}")

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
                            lat = float(payload.get("lat", 0.0))
                            lng = float(payload.get("lng", 0.0))
                            pois_data = buscar_nuevo_lugar("atracciones turisticas", lat, lng)

                            room_state["system_context_str"] = f"""
                            Eres Locus un guía turístico. Contexto: {user_ctx}
                            Latitud actual: {lat}. Longitud actual: {lng}.
                            Lugares iniciales: {pois_data}

                            REGLAS CRÍTICAS:
                            1. Cuando añadas POIS indica al usaurio que mire el mapa.
                            2. Si piden recomendaciones, usa la herramienta buscar_nuevo_lugar.
                            3. Las atracciones turisticas iniciales tienen que ser las principales que aparecen en todas las guias turisticas, después dejate guiar por el perfil del usuario.
                            4. No des la razón por darla, di la verdad técnica.
                            5. FORMATO OBLIGATORIO:
                            <POIS>[{{"name": "Nombre", "lat": 0.0, "lng": 0.0, "description": "Descripción"}}]</POIS>
                            6. VOZ Y ACENTO: Eres una entidad masculina. Tu acento por defecto DEBE ser Español de España (castellano peninsular). Si el 'Contexto' indica que el usuario es de otro país (ej. México, Inglaterra), adapta tu acento y tu idioma a su lugar de origen de forma completamente natural.
                            """
                            first_query = f"{room_state['system_context_str']}\nUsuario: Hola, acabo de llegar. Preséntate y dime qué hay cerca."
                            response = chat_session.send_message(first_query)
                            await manager.broadcast_text(response.text, room_id)
                            continue

                        if action == "start_voice_call":
                            poi_name = payload.get("poi_name", "tu destino")
                            user_text = f"El usuario acaba de iniciar la llamada y ha seleccionado {poi_name}. Salúdale por voz, recomienda cascos y pregunta si empezamos."
                            await process_voice_turn(user_text)
                            continue

                        if action == "text_chat":
                            user_text = payload.get("data")
                            logger.info(f"Texto del teclado recibido: {user_text}")
                            await process_voice_turn(user_text)
                            continue

                        if action == "audio_chat":
                            logger.info("Recibido audio del usuario. Iniciando transcripción...")
                            audio_b64 = payload.get("data")
                            mime_type = payload.get("mime_type", "audio/aac")
                            audio_bytes = base64.b64decode(audio_b64)
                            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

                            transcription = await client.aio.models.generate_content(
                                model=TEXT_MODEL,
                                contents=[audio_part, "Transcribe exactamente lo que dice este audio en español. Devuelve solo el texto sin comillas."]
                            )

                            user_text = transcription.text.strip()
                            logger.info(f"Transcripción obtenida: {user_text}")

                            if user_text:
                                await process_voice_turn(user_text)
                            continue

                        if action == "image_context":
                            logger.info("Procesando imagen con IA Vision...")
                            img_b64 = payload.get("data")
                            mime_type = payload.get("mime_type", "image/jpeg")
                            img_bytes = base64.b64decode(img_b64)

                            user_text = "Te estoy enseñando esto que estoy viendo ahora mismo. Tenlo en cuenta para la interacción y explícame qué es en tu rol de guía turístico."
                            await process_voice_turn(user_text, image_bytes=img_bytes, mime_type=mime_type)
                            continue

                    except Exception as e:
                        logger.error(f"Error procesando JSON: {e}")
                        continue

                response = chat_session.send_message(data)
                await manager.broadcast_text(response.text, room_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
    except Exception as e:
        logger.error(f"Error Critico: {e}")
        manager.disconnect(websocket, room_id)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)