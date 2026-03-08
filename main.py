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


class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id

        self.chat_session = client.chats.create(
            model=TEXT_MODEL,
            config=types.GenerateContentConfig(
                tools=[buscar_nuevo_lugar],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
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
        logger.info(f"[{room_id}] Cliente conectado. Total={len(self.active_connections[room_id])}")

    async def disconnect(self, websocket: WebSocket, room_id: str):
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


def chunk_audio_bytes(audio_bytes: bytes, chunk_size: int = AUDIO_CHUNK_SIZE):
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i:i + chunk_size]


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
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    logger.info(f"[{room_id}] Abriendo sesión Gemini Live con modelo {LIVE_MODEL}")
    logger.info(f"[{room_id}] Live config: AUDIO + input_audio_transcription + output_audio_transcription")

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
            room_state.live_ready.set()
            logger.info(f"[{room_id}] Sesión Live lista")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        sc = getattr(response, "server_content", None)
                        if sc is None:
                            logger.info(f"[{room_id}] Evento Live sin server_content: {response}")
                            continue

                        input_transcription = getattr(sc, "input_transcription", None)
                        if input_transcription is not None:
                            input_text = getattr(input_transcription, "text", "") or ""
                            input_text = input_text.strip()
                            if input_text:
                                logger.info(f"[{room_id}] Gemini entendió al usuario: {input_text[:120]}")

                        output_transcription = getattr(sc, "output_transcription", None)
                        if output_transcription is not None:
                            output_text = getattr(output_transcription, "text", "") or ""
                            output_text = output_text.strip()
                            if output_text:
                                logger.info(f"[{room_id}] Gemini -> transcripción: {output_text[:120]}")
                                await manager.broadcast_text(output_text, room_id)

                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and inline_data.data:
                                    data = inline_data.data
                                    logger.info(f"[{room_id}] Gemini -> audio bytes: {len(data)}")
                                    await manager.broadcast_bytes(data, room_id)

                                text_part = getattr(part, "text", None)
                                if text_part:
                                    text_part = text_part.strip()
                                    if text_part:
                                        logger.info(f"[{room_id}] Gemini -> text part: {text_part[:120]}")
                                        await manager.broadcast_text(text_part, room_id)

                        if sc.turn_complete:
                            logger.info(f"[{room_id}] Turno completado por Gemini")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"[{room_id}] Error recibiendo de Gemini: {e}")
                    await manager.broadcast_text(
                        "Ahora mismo no puedo responder por voz. Prueba otra vez.",
                        room_id
                    )

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
                            await session.send(input=text, end_of_turn=True)

                        elif payload_type == "audio_chunk":
                            audio_bytes = payload.get("data", b"")
                            is_last = payload.get("is_last", False)

                            if not audio_bytes:
                                continue

                            logger.info(
                                f"[{room_id}] -> Gemini audio chunk: {len(audio_bytes)} bytes | is_last={is_last}"
                            )
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
                    await manager.broadcast_text(
                        "Se ha cortado la conversación de voz. Inténtalo otra vez.",
                        room_id
                    )

            await asyncio.gather(receive_from_gemini(), send_to_gemini())

    except asyncio.CancelledError:
        logger.info(f"[{room_id}] Sesión Live cancelada")
        raise
    except Exception as e:
        logger.exception(f"[{room_id}] Error abriendo conexión Live: {e}")
        await manager.broadcast_text("No he podido abrir la sesión de voz.", room_id)
    finally:
        room_state.live_ready.clear()
        logger.info(f"[{room_id}] Sesión Live cerrada")


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

                        room_state.system_context_str = f"""
Eres Locus, un guía turístico experto, directo y conversacional.
Perfil de los viajeros: {user_ctx}.
Latitud actual: {lat}. Longitud actual: {lng}.
Lugares iniciales: {pois_data}

REGLAS CRÍTICAS DE COMPORTAMIENTO:
1. Ultra Brevedad: Tus respuestas habladas deben tener un máximo de 2 o 3 frases cortas. Eres un acompañante, no una enciclopedia.
2. Foco Espacial: Limita tus explicaciones ESTRICTAMENTE al lugar o monumento que el usuario está visitando en este momento.
3. Cero Coletillas: Nunca uses frases como '¡Qué interesante!' o 'Buena pregunta'. Ve directo al dato.
4. Voz y Acento: Eres una entidad masculina. Español de España por defecto.
5. FORMATO OBLIGATORIO DE POIS PARA EL MAPA:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
6. REGLA DE CONVERSACIÓN ADAPTATIVA:
Da un dato breve y termina cediendo el control al usuario.
""".strip()

                        first_query = (
                            f"{room_state.system_context_str}\n"
                            "Usuario: Hola, acabo de llegar. Preséntate y dime qué hay cerca."
                        )

                        response = room_state.chat_session.send_message(first_query)
                        await manager.broadcast_text(response.text, room_id)
                        continue

                    if action == "start_voice_call":
                        await ensure_live_session(room_id)

                        poi_name = payload.get("poi_name", "tu destino")
                        user_text = (
                            "INSTRUCCIONES DE COMPORTAMIENTO:\n"
                            f"{room_state.system_context_str}\n\n"
                            "SITUACIÓN ACTUAL:\n"
                            f"El usuario acaba de iniciar la ruta en {poi_name}. "
                            "En un máximo de 2 frases: dale la bienvenida a este lugar exacto, "
                            "recomiéndale ponerse los auriculares para aislarse del ruido, "
                            "y pregúntale hacia dónde está mirando."
                        )

                        await room_state.enqueue({
                            "type": "text",
                            "text": user_text,
                        })
                        continue

                    if action == "text_chat":
                        await ensure_live_session(room_id)

                        user_text = (payload.get("data") or "").strip()
                        if user_text:
                            await room_state.enqueue({
                                "type": "text",
                                "text": user_text,
                            })
                        continue

                    if action == "audio_stream":
                        await ensure_live_session(room_id)

                        try:
                            audio_b64 = payload.get("data", "")
                            audio_bytes = base64.b64decode(audio_b64)
                            if audio_bytes:
                                logger.info(f"[{room_id}] audio_stream bytes={len(audio_bytes)}")
                                await room_state.enqueue({
                                    "type": "audio_chunk",
                                    "data": audio_bytes,
                                    "is_last": False,
                                })
                        except Exception as e:
                            logger.exception(f"[{room_id}] Error en audio_stream: {e}")
                        continue

                    if action == "audio_end":
                        logger.info(f"[{room_id}] audio_end recibido (sin uso en v1)")
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

                            for i, chunk in enumerate(chunks):
                                await room_state.enqueue({
                                    "type": "audio_chunk",
                                    "data": chunk,
                                    "is_last": i == len(chunks) - 1,
                                })

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
                                "Identifica el elemento arquitectónico, cuadro o detalle que domina "
                                "la imagen y dispara directamente un dato curioso en una sola frase."
                            )

                            await room_state.enqueue({
                                "type": "image_turn",
                                "image_bytes": img_bytes,
                                "mime_type": mime_type,
                                "prompt": prompt,
                            })

                        except Exception as e:
                            logger.exception(f"[{room_id}] Error procesando image_context: {e}")
                        continue

                except Exception as e:
                    logger.exception(f"[{room_id}] Error procesando JSON: {e}")
                    continue

            try:
                response = room_state.chat_session.send_message(raw_data)
                await manager.broadcast_text(response.text, room_id)
            except Exception as e:
                logger.exception(f"[{room_id}] Error en chat_session: {e}")
                await manager.broadcast_text(
                    "No he podido procesar ese mensaje.",
                    room_id
                )

    except WebSocketDisconnect:
        await manager.disconnect(websocket, room_id)
    except Exception as e:
        logger.exception(f"[{room_id}] WebSocket Error: {e}")
        await manager.disconnect(websocket, room_id)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)