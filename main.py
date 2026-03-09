import os
import logging
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, multimodal
from livekit.plugins import gemini
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("LocusAgent")

# El System Prompt de Locus: su alma sigue siendo la misma
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

    # Inicializamos el modelo nativo de Gemini para LiveKit
    # No usamos gemini-2.5-flash-native-audio aquí, el plugin usa gemini-2.0-flash-exp 
    # que es el estándar actual para multimodal real-time en LiveKit.
    model = gemini.MultimodalAgent(
        model="gemini-2.0-flash-exp",
        api_key=os.environ.get("GEMINI_API_KEY"),
        instructions=SYSTEM_PROMPT,
    )

    # Creamos el agente multimodal (él maneja VAD, cancelación de eco e interrupciones)
    agent = multimodal.MultimodalAgent(model=model)

    # Arrancamos el agente en la sala
    agent.start(ctx.room)
    
    # Saludo inicial automático al entrar
    await agent.say("Hola, soy Locus. Ya estoy aquí para acompañarte en tu visita. ¿Qué es lo primero que tienes delante?", allow_interruptions=True)

if __name__ == "__main__":
    # LiveKit Agents no usa uvicorn, usa su propio CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))