import os
import logging
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import google

load_dotenv()
logger = logging.getLogger("LocusAgent")

SYSTEM_PROMPT = """
Eres Locus, un guía turístico experto, carismático y directo. Estás acompañando presencialmente a los viajeros en Albacete.

REGLAS DE ORO:
1. FOCO DE TÚNEL: Habla exclusivamente de lo que el usuario tiene delante.
2. STORYTELLING: No recites Wikipedia. Cuenta el secreto histórico, el detalle técnico o la leyenda apasionante.
3. ADAPTACIÓN: Ajusta tu vocabulario al tono del viajero.
4. BREVEDAD: Responde en un máximo de 2 o 3 frases.
5. ENGANCHE: Termina siempre con una pregunta directa sobre el monumento o detalle que están viendo.
"""

async def entrypoint(ctx: JobContext):
    logger.info(f"Conectando a la sala: {ctx.room.name}")
    await ctx.connect()

    # En LiveKit 1.0+, AgentSession es el cerebro unificado
    # Le pasamos nuestro modelo nativo intocable de Gemini
    session = AgentSession(
        llm=google.LLM(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-preview-12-2025"
        )
    )

    # Definimos las instrucciones (el rol)
    agent = Agent(instructions=SYSTEM_PROMPT)

    # Arrancamos el túnel WebRTC
    await session.start(agent=agent, room=ctx.room)

    # Saludo inicial para romper el hielo en cuanto se conecte el móvil
    await session.generate_reply(
        instructions="Hola, soy Locus. Ya estoy operativo. ¿Qué estamos viendo?"
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))