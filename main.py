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
                            "description": "Lugar cercano"
                        }
                    )

            return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.exception(f"Error en Maps: {e}")
        return "[]"

def build_home_context(user_ctx: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""Eres Locus, un guía turístico local empático, vibrante y muy cercano.
Perfil de los viajeros a los que guías: {user_ctx}
Coordenadas de tu posición: Latitud {lat}, longitud {lng}
Lugares a tu alrededor: {pois_data}

INSTRUCCIONES CLAVE:
1. ADAPTACIÓN CAMALEÓNICA: Ajusta tu vocabulario, tono y energía para que encaje perfectamente con el perfil de tus viajeros. Sé su amigo local experto.
2. Eres ágil y resolutivo. Nada de introducciones robóticas.
3. El formato obligatorio para listar los lugares en el mapa es estrictamente este:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
"""

def build_voice_context(user_ctx: str, lat: float, lng: float, pois_data: str, poi_name: str) -> str:
    ubicacion_foco = f"SITUACIÓN ABSOLUTA: Estás parado físicamente justo delante de {poi_name}." if poi_name else "SITUACIÓN: Caminando por la ciudad, sin destino fijo todavía."
    
    return f"""Eres Locus, el guía turístico más carismático, empático y observador de la ciudad. Estás haciendo un tour presencial, caminando codo a codo con los viajeros.

QUIÉN TE ESCUCHA (TU AUDIENCIA): {user_ctx}
{ubicacion_foco}
ENTORNO (Solo para que tú sepas la zona): {pois_data}

CÓMO DEBES COMPORTARTE SIEMPRE:
1. CONEXIÓN HUMANA: Mimetízate con la audiencia. Si son expertos, usa lenguaje sofisticado; si es un tono casual o joven, sé divertido y coloquial. Habla como una persona real, con calidez.
2. ANCLAJE FÍSICO INQUEBRANTABLE: Tú estás EN {poi_name}. Todo lo que digas debe referirse a lo que tienes delante. NUNCA pidas confirmación de dónde estás o en qué ciudad te encuentras. Tú lideras.
3. STORYTELLING: No recites Wikipedia. Cuenta el cotilleo histórico, el detalle arquitectónico oculto o la leyenda apasionante de este lugar exacto.
4. AGILIDAD EXTREMA: Habla en un máximo de 2 a 3 frases. 
5. PASA EL MICRÓFONO: Termina siempre tu intervención con una pregunta directa y conversacional ("¿Te has fijado en ese balcón?", "¿Qué crees que pasó aquí?") para cederles el turno de forma natural.
"""

def build_poi_research_prompt(user_ctx: str, poi_name: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""Busca en tu base de datos la historia viva, anécdotas y detalles arquitectónicos sobre: {poi_name} (Ubicado cerca de {pois_data}).
Perfil de los viajeros: {user_ctx}.
Devuelve viñetas muy breves con:
- Año y creador.
- El salseo histórico o la curiosidad más fascinante del lugar.
- Un detalle visual específico en el que el visitante deba fijarse.
Nada de texto de relleno, solo los datos clave para que el guía los use."""

def build_turn_prompt(user_message: str, current_poi_name: str, poi_research_summary: str, last_image_summary: str = "") -> str:
    ctx = f"ESTÁIS PARADOS EN: {current_poi_name}.\n" if current_poi_name else ""
    if poi_research_summary:
        ctx += f"TUS NOTAS MENTALES SOBRE ESTE LUGAR (Úsalas para inspirar tu respuesta, no las leas de forma literal):\n{poi_research_summary}\n"
    if last_image_summary:
        ctx += f"LO QUE EL VIAJERO TE ACABA DE ENSEÑAR O SEÑALAR:\n{last_image_summary}\n"

    return f"""{ctx}
Viajero: "{user_message}"

Responde como Locus. Recuerda: máximo 3 frases, tono adaptado a su perfil, cálido, anclado al lugar, y termina con una pregunta natural:"""

def ask_openai_chat(system_context: str, user_message: str, allow_web_search: bool = False) -> str:
    args = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_message},
        ],
    }
    response = openai_client.chat.completions.create(**args)
    return response.choices[0].message.content.strip()

def ask_openai_image(system_context: str, prompt: str, image_bytes: bytes, mime_type: str, allow_web_search: bool = False) -> str:
    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
    args = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_context},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    response = openai_client.chat.completions.create(**args)
    return response.choices[0].message.content.strip()

def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = FRONT_AUDIO_SAMPLE_RATE, channels: int = FRONT_AUDIO_CHANNELS, sample_width: int = FRONT_AUDIO_BYTES_PER_SAMPLE) -> bytes:
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
        speed=1.15,  # Velocidad aumentada para inyectar dinamismo y reducir latencia
    )
    return response.read()

def sanitize_for_voice(text: str) -> str:
    if not text:
        return text
    sanitized = re.sub(r'https?://\S+', '', text)
    sanitized = re.sub(r'www\.\S+', '', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.system_context_str = ""
        self.current_poi_name = ""
        self.poi_research_summary = ""
        self.last_image_summary = ""
        self.poi_research_task: Optional[asyncio.Task] = None
        self.profile = {
            "user_ctx": "",
            "lat": 0.0,
            "lng": 0.0,
            "pois_data": "[]",
        }

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

async def enrich_poi_context_in_background(room_state: RoomState):
    try:
        poi_name = room_state.current_poi_name or "el lugar actual"
        lat = float(room_state.profile.get("lat", 0.0))
        lng = float(room_state.profile.get("lng", 0.0))
        user_ctx = room_state.profile.get("user_ctx", "")
        pois_data = room_state.profile.get("pois_data", "[]")

        system_context = "Eres un documentalista de élite. Tu objetivo es nutrir al guía con el contexto histórico más interesante."
        prompt = build_poi_research_prompt(user_ctx, poi_name, lat, lng, pois_data)

        summary = await asyncio.to_thread(
            ask_openai_chat,
            system_context,
            prompt,
            False,
        )
        room_state.poi_research_summary = summary.strip()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(f"[{room_state.room_id}] Error enriqueciendo POI: {e}")

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
        lat=float(room_state.profile.get("lat", 0.0)),
        lng=float(room_state.profile.get("lng", 0.0)),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=room_state.current_poi_name,
    )

    prompt = build_turn_prompt(
        user_message=user_message,
        current_poi_name=room_state.current_poi_name,
        poi_research_summary=room_state.poi_research_summary,
        last_image_summary=room_state.last_image_summary,
    )

    return await asyncio.to_thread(
        ask_openai_chat,
        system_context,
        prompt,
        False,
    )

async def respond_in_call(room_state: RoomState, room_id: str, user_message: str):
    answer = await generate_fast_answer(room_state, user_message)
    if not answer:
        answer = "No te he escuchado bien, ¿qué decías de este lugar?"
    await speak_text(room_state, room_id, answer)

async def respond_to_image_in_call(room_state: RoomState, room_id: str, image_bytes: bytes, mime_type: str):
    poi_name = room_state.current_poi_name or "el lugar actual"

    system_context = build_voice_context(
        user_ctx=room_state.profile.get("user_ctx", ""),
        lat=float(room_state.profile.get("lat", 0.0)),
        lng=float(room_state.profile.get("lng", 0.0)),
        pois_data=room_state.profile.get("pois_data", "[]"),
        poi_name=poi_name,
    )

    prompt = f"El visitante te acaba de señalar este detalle exacto en {poi_name}. Reacciona con naturalidad y cuéntale qué es en 2 frases, revelando un secreto interesante."

    answer = await asyncio.to_thread(
        ask_openai_image,
        system_context,
        prompt,
        image_bytes,
        mime_type,
        False,
    )

    if not answer:
        answer = "La perspectiva no me deja verlo bien, ¿qué te llama la atención exactamente?"

    room_state.last_image_summary = answer
    await speak_text(room_state, room_id, answer)

def infer_language_hint(user_ctx: str) -> Optional[str]:
    lowered = (user_ctx or "").lower()
    if "english" in lowered or "inglés" in lowered or "ingles" in lowered: return "en"
    if "italiano" in lowered or "italian" in lowered: return "it"
    if "francés" in lowered or "frances" in lowered or "french" in lowered: return "fr"
    return "es"

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
                            "Hola, acabamos de llegar. Da la bienvenida con entusiasmo, adáptate a nuestro perfil y dime qué hay cerca. Recuerda el bloque POIS.",
                            False,
                        )
                        await manager.broadcast_text(response_text, room_id)
                        continue

                    if action == "start_voice_call":
                        room_state.current_poi_name = payload.get("poi_name", "tu destino")
                        room_state.last_image_summary = ""

                        if room_state.poi_research_task and not room_state.poi_research_task.done():
                            room_state.poi_research_task.cancel()

                        room_state.poi_research_task = asyncio.create_task(
                            enrich_poi_context_in_background(room_state)
                        )

                        await respond_in_call(
                            room_state,
                            room_id,
                            f"Acabamos de llegar justo frente a {room_state.current_poi_name}. Reacciona al entorno, arranca el tour con un dato intrigante y pregúntame qué opino."
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
                    raw_data,
                    False,
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