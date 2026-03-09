import os
import logging
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Importaciones de LiveKit
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import google as livekit_google
from livekit import api

load_dotenv()
logger = logging.getLogger("LocusSystem")

# ==========================================
# 0. CONFIGURACIÓN IA TEXTO (PARA EL HOME)
# ==========================================
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
text_model = genai.GenerativeModel('gemini-2.5-flash')

# Memoria temporal para el chat de texto del Home
chat_histories = {}

# ==========================================
# 1. FASTAPI: EL SERVIDOR WEB (FRONTEND)
# ==========================================
app = FastAPI(title="Locus API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    action: str
    roomId: str
    deviceId: str
    context: str = ""
    text: str = ""
    lat: float = None
    lng: float = None

@app.post("/home_chat")
async def home_chat(req: ChatRequest):
    """
    Gestiona el chat de texto del Home y devuelve los POIs para el mapa.
    """
    if req.roomId not in chat_histories:
        chat_histories[req.roomId] = []

    history = chat_histories[req.roomId]

    if req.action == "setup_profile":
        prompt = f"""
        Eres Locus, un guía turístico experto en Albacete. El usuario está configurando su ruta.
        Sus coordenadas son: latitud {req.lat}, longitud {req.lng}.
        Su contexto/preferencias: '{req.context}'.

        Salúdale amablemente y sugiérele 3 sitios (POIs) cercanos o relevantes.
        IMPORTANTE: Tu respuesta debe terminar OBLIGATORIAMENTE con este bloque exacto (sin comillas invertidas extra):
        <POIS>
        [
          {{"name": "Nombre del sitio", "lat": 38.99, "lng": -1.85, "description": "Breve descripción"}}
        ]
        </POIS>
        """
        history.append({"role": "user", "parts": [prompt]})
    else:
        prompt = f"""
        El usuario dice: '{req.text}'.
        Responde como Locus. Si en tu respuesta sugieres sitios nuevos, añade el bloque <POIS> al final.
        Si no hay sitios nuevos, responde solo con texto.
        """
        history.append({"role": "user", "parts": [prompt]})

    # Llamamos a Gemini
    response = text_model.generate_content(history)
    bot_reply = response.text

    # Guardamos la respuesta en memoria
    history.append({"role": "model", "parts": [bot_reply]})

    return {"reply": bot_reply}

class TokenRequest(BaseModel):
    participant_name: str
    room_name: str
    poi_context: str = ""

@app.post("/get_token")
async def get_token(req: TokenRequest):
    """
    Genera el pase VIP para que el móvil pueda entrar a la sala WebRTC de LiveKit.
    """
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(req.participant_name)
    token.with_name(req.participant_name)
    token.with_metadata(req.poi_context) # Inyectamos datos

    grant = api.VideoGrant(room_join=True, room=req.room_name)
    token.with_grant(grant)

    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}


# ==========================================
# 2. LIVEKIT AGENT: EL CEREBRO DE VOZ (CALL)
# ==========================================
SYSTEM_PROMPT = """
Eres Locus, un guía turístico experto, carismático y directo. Estás acompañando presencialmente a los viajeros.
REGLAS:
1. FOCO DE TÚNEL: Habla exclusivamente de lo que el usuario tiene delante.
2. BREVEDAD: Responde en un máximo de 2 o 3 frases.
3. ENGANCHE: Termina con una pregunta directa sobre el monumento.
"""

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"🎙️ Locus conectado a la sala de voz: {ctx.room.name}")

    # Rescatamos contexto si lo hay
    user_context = ""
    for _, participant in ctx.room.remote_participants.items():
        if participant.metadata:
            user_context = participant.metadata
            break

    dynamic_prompt = SYSTEM_PROMPT
    if user_context:
        dynamic_prompt += f"\nATENCIÓN: El usuario está viendo actualmente: {user_context}."

    session = AgentSession(
        llm=livekit_google.LLM(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-preview-12-2025"
        )
    )

    agent = Agent(instructions=dynamic_prompt)
    await session.start(agent=agent, room=ctx.room)

    await session.generate_reply(
        instructions="El usuario acaba de entrar a la llamada de voz. Saluda de forma natural y pregúntale qué le parece lo que tiene delante."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))