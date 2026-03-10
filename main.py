import os
import json
import requests
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import google as livekit_google
from livekit import api

load_dotenv()

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

chat_histories = {}

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

def get_real_pois(query, lat, lng):
    api_key = os.environ.get("MAPS_API_KEY")
    if not api_key:
        return []
        
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": api_key
    }
    if lat and lng:
        params["location"] = f"{lat},{lng}"
        params["radius"] = 2000

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return []
        
    results = resp.json().get("results", [])[:3]
    pois = []
    
    for r in results:
        pois.append({
            "name": r.get("name"),
            "lat": r["geometry"]["location"]["lat"],
            "lng": r["geometry"]["location"]["lng"],
            "description": r.get("formatted_address", "")
        })
    return pois

@app.post("/home_chat")
async def home_chat(req: ChatRequest):
    if req.roomId not in chat_histories:
        chat_histories[req.roomId] = {"history": [], "lat": None, "lng": None}
        
    room_data = chat_histories[req.roomId]
    
    if req.lat is not None and req.lng is not None:
        room_data["lat"] = req.lat
        room_data["lng"] = req.lng
        
    history = room_data["history"]
    current_lat = room_data["lat"]
    current_lng = room_data["lng"]
    
    pois_block = ""
    
    if req.action == "setup_profile":
        real_pois = get_real_pois("lugares turísticos", current_lat, current_lng)
        nombres_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else "lugares cercanos"
        
        prompt = f"Eres Locus, un guía experto. El usuario configura su ruta. Su contexto/petición es: '{req.context}'. IGNORA cualquier ciudad en su contexto y recomiéndale ÚNICAMENTE estos lugares reales a su alrededor: {nombres_pois}. Salúdale asumiendo su rol, anclado a su ubicación."
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        
        if real_pois:
            pois_block = f"\n<POIS>\n{json.dumps(real_pois, ensure_ascii=False)}\n</POIS>"
    else:
        real_pois = get_real_pois(req.text, current_lat, current_lng)
        nombres_pois = ", ".join([p["name"] for p in real_pois]) if real_pois else ""
        
        prompt = f"El usuario dice: '{req.text}'."
        if real_pois:
            prompt += f" Lugares reales encontrados cerca: {nombres_pois}. Si tu respuesta sugiere lugares, debes incluir el bloque <POIS> exacto al final."
            pois_block = f"\n<POIS>\n{json.dumps(real_pois, ensure_ascii=False)}\n</POIS>"
        else:
            prompt += " Responde como Locus de forma concisa."
            
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=history
    )
    bot_reply = response.text
    
    if pois_block and "<POIS>" not in bot_reply:
        bot_reply += pois_block
        
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=bot_reply)]))

    return {"reply": bot_reply}

class TokenRequest(BaseModel):
    participant_name: str
    room_name: str
    poi_context: str = ""

@app.post("/get_token")
async def get_token(req: TokenRequest):
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(req.participant_name)
    token.with_name(req.participant_name)
    token.with_metadata(req.poi_context)
    
    grant = api.VideoGrant(room_join=True, room=req.room_name)
    token.with_grant(grant)
    
    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}

SYSTEM_PROMPT = """
Eres Locus, un guía turístico experto y directo. Estás acompañando presencialmente a los viajeros.

REGLAS ESTRICTAS DE COMPORTAMIENTO:
1. CERO PAJA: Tienes estrictamente prohibido usar frases de relleno. Tu primera palabra debe ser ya información útil.
2. RIGOR HISTÓRICO: No inventes absolutamente nada.
3. ANCLAJE ESPACIAL EXTREMO: Tienes PROHIBIDO mencionar o sugerir lugares cercanos, calles adyacentes u otros monumentos. Tu conocimiento está bloqueado y limitado EXCLUSIVAMENTE al monumento exacto que el usuario tiene delante. Si te preguntan por otro sitio, desvía la conversación de vuelta al monumento actual.
4. RESPUESTAS CORTAS: Tu respuesta no puede durar más de 2 o 3 frases.
5. ENGANCHE: Termina tu intervención con una pregunta corta y directa sobre lo que el usuario está viendo físicamente en ese momento.
"""

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    user_context = ""
    for _, participant in ctx.room.remote_participants.items():
        if participant.metadata:
            user_context = participant.metadata
            break

    dynamic_prompt = SYSTEM_PROMPT
    welcome_msg = "El usuario acaba de entrar a la llamada de voz. Saluda de forma natural."
    
    if user_context:
        dynamic_prompt += f"\nATENCIÓN: El usuario está viendo actualmente: {user_context}."
        welcome_msg = f"El usuario acaba de llegar a este lugar: {user_context}. Dale una bienvenida específica a este sitio y pregúntale qué le parece."

    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            instructions=dynamic_prompt,
            voice="Puck"
        )
    )
    
    agent = Agent(instructions=dynamic_prompt)
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply(
        instructions=welcome_msg
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))