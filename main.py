import os
import json
import re
import urllib.parse
import base64
import asyncio
import requests
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm
from livekit.plugins import google as livekit_google
from livekit import api, rtc

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
    enriched_context = req.poi_context
    match = re.search(r'Viendo:\s*(.*?)\.\s*Detalles', req.poi_context)

    if match:
        poi_name = match.group(1)
        try:
            resp = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Actúa como enciclopedia. Dame 2 datos reales y verificados (año exacto de inauguración/construcción y estilo arquitectónico o autor) sobre '{poi_name}' en España. Sé muy breve. Si no tienes el dato exacto 100% seguro, responde solo 'NO_DATA'."
            )
            if "NO_DATA" not in resp.text:
                enriched_context += f" | DATOS HISTÓRICOS REALES PARA QUE LOS USES: {resp.text}"
        except:
            pass

    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET")
    )
    token.with_identity(req.participant_name)
    token.with_name(req.participant_name)
    token.with_metadata(enriched_context)

    grant = api.VideoGrant(room_join=True, room=req.room_name)
    token.with_grant(grant)

    return {"token": token.to_jwt(), "ws_url": os.getenv("LIVEKIT_URL")}

class GuideTools(llm.FunctionContext):
    @llm.ai_callable(description="Busca información histórica o arquitectónica exacta de un monumento si el usuario hace una pregunta que no sabes responder con seguridad.")
    async def buscar_en_internet(self, monumento: str):
        url = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(monumento)}&utf8=&format=json"
        try:
            resp = requests.get(url, timeout=3).json()
            if resp.get("query", {}).get("search"):
                snippet = resp["query"]["search"][0]["snippet"]
                snippet_limpio = re.sub(r'<[^>]+>', '', snippet)
                return f"Información verificada encontrada: {snippet_limpio}"
        except:
            pass
        return "No se ha encontrado información adicional en las bases de datos externas."

SYSTEM_PROMPT = """
Eres Locus, un guía turístico experto, carismático y directo que acompaña presencialmente al usuario.

REGLAS DE ORO:
1. CERO PAJA: Prohibido usar frases de relleno. Empieza directo con la información útil.
2. RIGOR HISTÓRICO ABSOLUTO: NUNCA inventes fechas, siglos ni nombres.
3. ANCLAJE ESPACIAL: Habla SOLO del monumento en el que está el usuario ahora mismo.
4. CONCISIÓN: Respuestas de 2 frases como máximo.
5. ENGANCHE VISUAL: Termina con una pregunta breve sobre algún detalle físico.
6. BÚSQUEDA EN VIVO (HERRAMIENTA): Tienes acceso a la herramienta 'buscar_en_internet'. Úsala SOLO si te preguntan un dato específico que no tienes en tu memoria ni en tu contexto. IMPORTANTE: Cuando decidas usar la herramienta, primero debes decirle en voz alta al usuario algo como "Dame un segundo que lo consulto..." o "Espera que reviso mis notas...", y seguidamente ejecutas la herramienta.
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
        dynamic_prompt += f"\nCONTEXTO DEL USUARIO: {user_context}."
        welcome_msg = f"El usuario acaba de llegar a este lugar: {user_context}. Dale una bienvenida específica a este sitio usando los DATOS HISTÓRICOS REALES si los hay en tu contexto, y pregúntale qué le parece visualmente."

    fnc_ctx = GuideTools()

    session = AgentSession(
        llm=livekit_google.beta.realtime.RealtimeModel(
            api_key=os.environ.get("GEMINI_API_KEY"),
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            instructions=dynamic_prompt,
            voice="Puck"
        )
    )

    agent = Agent(instructions=dynamic_prompt, fnc_ctx=fnc_ctx)
    await session.start(agent=agent, room=ctx.room)

    @ctx.room.on("data_received")
    def on_data_received(data_packet: rtc.DataPacket):
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
            if payload.get("action") == "text_chat":
                asyncio.create_task(session.generate_reply(user_input=payload["data"]))
            elif payload.get("action") == "image_context":
                mime_type = payload.get("mime_type", "image/jpeg")
                image_bytes = base64.b64decode(payload["data"])

                async def process_image():
                    try:
                        def fetch_desc():
                            return gemini_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                    types.Part.from_text(text="Describe de forma concisa lo que se ve en esta imagen, centrándote en el aspecto arquitectónico o turístico si lo hay.")
                                ]
                            ).text

                        descripcion = await asyncio.to_thread(fetch_desc)
                        await session.generate_reply(instructions=f"El usuario te acaba de enseñar una foto por el chat. Esto es lo que se ve en ella: {descripcion}. Haz un comentario breve y natural sobre la foto como guía.")
                    except Exception:
                        pass

                asyncio.create_task(process_image())
        except Exception:
            pass

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        welcome_str = "Un nuevo usuario se ha unido a la llamada."
        if participant.metadata:
            welcome_str += f" Está viendo este lugar: {participant.metadata}. Dale la bienvenida a este monumento de forma natural."
        asyncio.create_task(session.generate_reply(instructions=welcome_str))

    await session.generate_reply(
        instructions=welcome_msg
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))