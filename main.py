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
        chat_histories[req.roomId] = []
        
    history = chat_histories[req.roomId]
    pois_block = ""
    
    if req.action == "setup_profile":
        prompt = f"Eres Locus, un guía experto. El usuario configura su ruta. Contexto: '{req.context}'. Salúdale de forma breve y amigable."
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        
        real_pois = get_real_pois(f"lugares turisticos {req.context}", req.lat, req.lng)
        if real_pois:
            pois_block = f"\n<POIS>\n{json.dumps(real_pois, ensure_ascii=False)}\n</POIS>"
    else:
        prompt = f"El usuario dice: '{req.text}'. Responde como Locus de forma concisa."
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=history
    )
    bot_reply = response.text
    
    if pois_block:
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
Eres Locus, un guía turístico experto, carismático y directo. Estás acompañando presencialmente a los viajeros.
REGLAS:
1. FOCO DE TÚNEL: Habla exclusivamente de lo que el usuario tiene delante.
2. BREVEDAD: Responde en un máximo de 2 o 3 frases.
3. ENGANCHE: Termina con una pregunta directa sobre el monumento.
"""

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    user_context = ""
    for _, participant in ctx.room.remote_participants.items():
        if participant.metadata:
            user_context = participant.metadata
            break

    dynamic_prompt = SYSTEM_PROMPT
    if user_context:
        dynamic_prompt += f"\nATENCIÓN: El usuario está viendo actualmente: {user_context}."

    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            instructions=dynamic_prompt,
            voice="Puck"
        )
    )
    
    agent = Agent()
    await session.start(agent=agent, room=ctx.room)
    
    await session.generate_reply(
        instructions="El usuario acaba de entrar a la llamada de voz. Saluda de forma natural y pregúntale qué le parece lo que tiene delante."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))