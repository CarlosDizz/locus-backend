import os
import logging
import asyncio
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
# EL FIX DEL ZASCA: Importamos la clase directamente desde el submódulo
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import google
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("LocusAgent")

SYSTEM_PROMPT = """
Eres Locus, un guía turístico experto, carismático y directo. Estás acompañando presencialmente a los viajeros.

REGLAS DE ORO:
1. FOCO DE TÚNEL: Habla exclusivamente de lo que el usuario tiene delante.
2. STORYTELLING: No recites Wikipedia. Cuenta el secreto histórico, el detalle técnico o la leyenda apasionante.
3. ADAPTACIÓN: Ajusta tu vocabulario al tono del viajero.
4. BREVEDAD: Responde en un máximo de 2 o 3 frases.
5. ENGANCHE: Termina siempre con una pregunta directa sobre el monumento o detalle que están viendo.
"""

async def entrypoint(ctx: JobContext):
    logger.info(f"Conectando a la sala: {ctx.room.name}")
    
    # LÍNEA CRÍTICA NUEVA: El worker tiene que "entrar" físicamente a la sala
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Inicializamos el modelo de Google con nuestro cerebro nativo de audio
    model = google.Gemini(
        api_key=os.environ.get("GEMINI_API_KEY"),
        instructions=SYSTEM_PROMPT,
        model="gemini-2.5-flash-native-audio-preview-12-2025"
    )

    # Instanciamos el agente usando la clase importada correctamente
    agent = MultimodalAgent(model=model)

    # Arrancamos
    agent.start(ctx.room)
    
    # Saludo inicial
    await agent.say("Hola, soy Locus. Ya estoy operativo. ¿Qué estamos viendo?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))