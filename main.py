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
Perfil de los viajeros: {user_ctx}
Coordenadas: Lat {lat}, Lng {lng}
Lugares cercanos: {pois_data}

1. Adapta tu vocabulario al perfil. Eres ágil y resolutivo.
2. El formato para listar los lugares en el mapa es estrictamente este:
<POIS>[{{"name":"Nombre","lat":0.0,"lng":0.0,"description":"Descripción"}}]</POIS>
"""

def build_voice_context(user_ctx: str, lat: float, lng: float, pois_data: str, poi_name: str, research: str, last_img: str) -> str:
    ubicacion_foco = f"ESTÁS FÍSICAMENTE EN: {poi_name}." if poi_name else "Caminando por la ciudad, sin destino fijo."
    
    info = ""
    if research:
        info += f"\nTUS NOTAS DE EXPERTO (No las leas literales, úsalas para dar datos profundos):\n{research}\n"
    if last_img:
        info += f"\nÚLTIMO DETALLE SEÑALADO POR EL USUARIO:\n{last_img}\n"

    return f"""Eres Locus, el guía turístico más carismático y observador. Estás acompañando presencialmente a estos viajeros.

PERFIL DE TU AUDIENCIA: {user_ctx}
{ubicacion_foco}
{info}

REGLAS DE ORO:
1. MEMORIA Y CONTINUIDAD: Tienes en cuenta todo lo que habéis hablado antes. Si el usuario habla de un elemento concreto (una locomotora, una puerta, una estatua), MANTÉN EL TEMA en ese detalle. No vuelvas a darle la chapa con el contexto general del lugar.
2. PROFUNDIDAD RADICAL: No repitas datos. Si ya has presentado el monumento, baja al detalle. Cuéntale un secreto arquitectónico, una anécdota oscura o un fallo de construcción.
3. CONEXIÓN: Adapta tu acento y energía al usuario. Nunca pidas confirmar en qué ciudad estáis.
4. EXTREMA BREVEDAD: Responde en 2 o 3 frases como máximo.
5. PASA EL MICRÓFONO: Termina siempre preguntándoles su opinión sobre el detalle exacto que estáis mirando.
"""

def build_poi_research_prompt(user_ctx: str, poi_name: str, lat: float, lng: float, pois_data: str) -> str:
    return f"""Eres el documentalista de un guía experto. Busca la historia, anécdotas avanzadas y detalles técnicos visuales sobre: {poi_name} (Zona: {pois_data}).
Perfil de la audiencia: {user_ctx}.
Devuelve solo viñetas rápidas con datos duros: año de creación, creador, un secreto histórico o leyenda, y 2 detalles en los que fijarse in situ. Sin relleno."""

def ask_openai_chat(system_context: str, user_message: str, history: list = None) -> str:
    messages = [{"role": "system", "content": system_context}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    args = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
    }
    response = openai_client.chat.completions.create(**args)
    return response.choices[0].message.content.strip()

def ask_openai_image(system_context: str, prompt: str, image_bytes: bytes, mime_type: str, history: list = None) -> str:
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

    args = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
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

class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.system_context_str = ""
        self.current_poi_name = ""
        self.poi_research_summary = ""
        self.last_image_summary = ""
        self.chat_history = [] 
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

        prompt = build_poi_research_prompt(user_ctx, poi_name, lat, lng, pois_data)
        summary = await asyncio.to_thread(
            ask_openai_chat,
            "Eres un documentalista de élite. Ve al grano.",
            prompt
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
        research=room_state.poi_research_summary,
        last_img=room_state.last_image_summary
    )

    prompt = f"Continúa la conversación con total naturalidad.\nViajero: \"{user_message}\""

    answer = await asyncio.to_thread(
        ask_openai_chat,
        system_context,
        prompt,
        room_state.chat_history
    )

    # Inyección de memoria: guardamos las últimas 12 frases para que recuerde el hilo exacto
    room_state.chat_history.append({"role": "user", "content": user_message})
    room_state.chat_history.append({"role": "assistant", "content": answer})
    room_state.chat_history = room_state.chat_history[-12:]

    return answer

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
        research=room_state.poi_research_summary,
        last_img=""
    )

    prompt = f"El visitante te acaba de señalar este detalle exacto en {poi_name}. Reacciona y cuéntale qué es en 2 frases, aportando un dato experto."

    answer = await asyncio.to_thread(
        ask_openai_image,
        system_context,
        prompt,
        image_bytes,
        mime_type,
        room_state.chat_history
    )

    if not answer:
        answer = "La perspectiva no me deja verlo bien, ¿qué te llama la atención exactamente?"

    room_state.last_image_summary = answer
    
    room_state.chat_history.append({"role": "user", "content": "[El usuario te ha enseñado una foto]"})
    room_state.chat_history.append({"role": "assistant", "content": answer})
    room_state.chat_history = room_state.chat_history[-12:]

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
                            "Hola, acabamos de llegar. Da la bienvenida con entusiasmo y dime qué hay cerca. Usa el bloque POIS."
                        )
                        await manager.broadcast_text(response_text, room_id)
                        continue

                    if action == "start_voice_call":
                        room_state.current_poi_name = payload.get("poi_name", "tu destino")
                        room_state.last_image_summary = ""
                        room_state.chat_history = [] # Limpiamos memoria al empezar un POI nuevo

                        if room_state.poi_research_task and not room_state.poi_research_task.done():
                            room_state.poi_research_task.cancel()

                        room_state.poi_research_task = asyncio.create_task(
                            enrich_poi_context_in_background(room_state)
                        )

                        await respond_in_call(
                            room_state,
                            room_id,
                            f"Acabamos de llegar justo frente a {room_state.current_poi_name}. Reacciona al entorno y arranca con un dato intrigante sobre la fachada o estructura."
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

            # Fallback para chat plano (mapa)
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