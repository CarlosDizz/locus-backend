import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Importaciones de LiveKit
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import google
from livekit import api

load_dotenv()
logger = logging.getLogger("LocusSystem")

# ==========================================
# 1. FASTAPI: EL SERVIDOR WEB (FRONTEND)
# ==========================================
app = FastAPI(title="Locus API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Tus POIs base para que el Home no casque
POIS = [
    {"id": "1", "name": "Catedral de San Juan Bautista", "description": "Catedral gótica con pinturas murales."},
    {"id": "2", "name": "Pasaje de Lodares", "description": "Galería comercial modernista del siglo XX."},
    {"id": "3", "name": "Recinto Ferial", "description": "Edificio conocido como 'La Sartén'."}
]

@app.get("/pois")
async def get_pois():
    return POIS

@app.get("/chat/history")
async def get_chat_history():
    return [] # Aquí irá tu lógica de base de datos en el futuro

class TokenRequest(BaseModel):
    participant_name: str # ej: tu deviceId
    room_name: str        # ej: Albacete_Centro
    poi_context: str = "" # ej: "Está viendo el Pasaje de Lodares"

@app.post("/get_token")
async def get_token(req: TokenRequest):
    """
    Este endpoint es vital: Ionic lo llama antes de conectar.
    Aquí le metemos las variables del perfil al token de LiveKit.
    """
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(req.participant_name)
    token.with_name(req.participant_name)
    # INYECCIÓN DE VARIABLES: Guardamos el POI en los metadatos del usuario
    token.with_metadata(req.poi_context)

    grant = api.VideoGrant(room_join=True, room=req.room_name)
    token.with_grant(grant)

    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}


# ==========================================
# 2. LIVEKIT AGENT: EL CEREBRO DE VOZ
# ==========================================
SYSTEM_PROMPT = """
Eres Locus, un guía turístico experto, carismático y directo.
REGLAS:
1. FOCO DE TÚNEL: Habla exclusivamente de lo que el usuario tiene delante.
2. BREVEDAD: Responde en un máximo de 2 o 3 frases.
3. ENGANCHE: Termina con una pregunta directa.
"""

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"🎙️ Locus conectado a la sala: {ctx.room.name}")

    # MAGIA: Rescatamos las variables que FastAPI metió en el token
    user_context = ""
    for _, participant in ctx.room.remote_participants.items():
        if participant.metadata:
            user_context = participant.metadata
            break

    # Adaptamos el cerebro de Gemini al POI actual
    dynamic_prompt = SYSTEM_PROMPT
    if user_context:
        dynamic_prompt += f"\nATENCIÓN: El usuario está viendo actualmente: {user_context}."

    session = AgentSession(
        llm=google.LLM(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-preview-12-2025"
        )
    )

    agent = Agent(instructions=dynamic_prompt)
    await session.start(agent=agent, room=ctx.room)

    await session.generate_reply(
        instructions="Saluda de forma natural y pregúntale qué le parece lo que tiene delante."
    )

if __name__ == "__main__":
    # Si ejecutamos 'python main.py start', arranca el agente de voz
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))