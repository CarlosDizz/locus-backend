from app.clients.openai_client import OpenAIClient, OpenAIClientError
from app.config import settings
from app.schemas.realtime import (
    RealtimeClientSecretResponse,
    RealtimePhotoInsightRequest,
    RealtimePhotoInsightResponse,
    RealtimeSessionRequest,
    RealtimeSessionResponse,
)
from app.services.billing_service import billing_service
from app.services.poi_service import poi_service
from app.services.prompt_service import prompt_service
from app.services.session_service import session_service
from app.tools.knowledge_tools import get_knowledge_tool_manifest
from app.tools.poi_tools import get_poi_tool_manifest
from app.tools.session_tools import get_session_tool_manifest


class RealtimeService:
    WEBRTC_CALLS_API_URL = "https://api.openai.com/v1/realtime/calls"

    def __init__(self) -> None:
        self.openai = OpenAIClient()

    def prepare_session(self, data: RealtimeSessionRequest) -> RealtimeSessionResponse:
        if data.user_id is not None:
            session_service.attach_user(data.session_id, data.user_id)
        session = session_service.get_or_create(data.session_id)
        active_poi_name = data.active_poi_name or (session.active_poi.name if session.active_poi else "")
        visit_context = data.visit_context or ""
        if active_poi_name:
            try:
                summary = poi_service.get_poi_summary(active_poi_name)
                if summary:
                    visit_context = summary
            except Exception:
                pass
        instructions = prompt_service.render(
            "realtime_agent.json",
            {
                "session_profile": session.profile.raw_context,
                "active_poi": active_poi_name,
                "visit_context": visit_context,
                "recent_memory": "\n".join(
                    f"{item['role'].upper()}: {item['text']}" for item in session.memory[-8:]
                ),
            },
        )
        raw_tools = [
            *get_session_tool_manifest(),
            *get_poi_tool_manifest(),
            *get_knowledge_tool_manifest(include_web_research_tool=True),
        ]
        tools = [{k: v for k, v in tool.items() if k != "strict"} for tool in raw_tools]
        return RealtimeSessionResponse(
            session_id=session.session_id,
            model=self.openai.realtime_model(),
            transport="webrtc",
            webrtc_api_url=self.WEBRTC_CALLS_API_URL,
            instructions=instructions,
            modalities=["text", "audio", "image"],
            tools=tools,
        )

    def create_client_secret(self, data: RealtimeSessionRequest) -> RealtimeClientSecretResponse:
        prepared = self.prepare_session(data)
        session = session_service.get_or_create(data.session_id)
        if session.user_id is not None:
            billing_service.ensure_user_can_consume(session.user_id)
        if not self.openai.is_configured():
            raise OpenAIClientError("OPENAI_API_KEY no está configurada")

        payload = self.openai.create_realtime_client_secret(
            model=prepared.model,
            instructions=prepared.instructions,
            modalities=["audio"],
            tools=prepared.tools,
            voice=self.openai.realtime_voice(),
            max_output_tokens=settings.openai_realtime_max_output_tokens,
        )
        client_secret = payload.get("client_secret", {}) or payload.get("session", {}).get("client_secret", {})
        expires_at = client_secret.get("expires_at") or payload.get("expires_at") or 0
        value = client_secret.get("value") or payload.get("value") or ""
        return RealtimeClientSecretResponse(
            session_id=prepared.session_id,
            model=prepared.model,
            client_secret=value,
            expires_at=expires_at,
            webrtc_api_url=self.WEBRTC_CALLS_API_URL,
            instructions=prepared.instructions,
            voice=self.openai.realtime_voice(),
            modalities=["audio"],
            tools=prepared.tools,
        )

    def analyze_photo(self, data: RealtimePhotoInsightRequest) -> RealtimePhotoInsightResponse:
        session = session_service.get_or_create(data.session_id)
        if session.user_id is not None:
            billing_service.ensure_user_can_consume(session.user_id)
        active_poi_name = session.active_poi.name if session.active_poi else ""
        nearby_names = ", ".join(poi.name for poi in session.nearby_pois[:6])
        ephemeral_names = ", ".join(poi.name for poi in session.ephemeral_map_pois[:4])
        memory = "\n".join(
            f"{item['role'].upper()}: {item['text']}" for item in session.memory[-6:]
        )

        instructions = "\n".join(
            [
                "Eres Locus, el guia de una llamada en vivo.",
                "Estas reaccionando a una foto enviada durante la visita.",
                "Responde en espanol natural y breve, como un mensaje de chat de 2 a 4 frases.",
                "Describe con claridad lo que ves.",
                "Si reconoces el lugar, monumento, obra o detalle urbano, identificalo con seguridad prudente.",
                "Si no puedes identificarlo con suficiente confianza, di lo que ves y pide un unico detalle corto para afinar.",
                "Usa el contexto de la visita y el POI activo solo si realmente encajan con la imagen.",
                "No inventes datos y no hables de modelos, tools, backend ni arquitectura interna.",
            ]
        )

        context_text = "\n".join(
            [
                f"POI activo: {active_poi_name or '(sin foco activo)'}",
                f"Lugares visibles en el mapa: {nearby_names or '(sin lugares visibles)'}",
                f"Recomendaciones efimeras marcadas: {ephemeral_names or '(sin recomendaciones efimeras)'}",
                f"Memoria reciente:\n{memory or '(sin memoria reciente)'}",
                f"Nombre del archivo: {data.file_name or '(sin nombre)'}",
            ]
        )

        payload = self.openai.create_response(
            model=self.openai.chat_model(),
            instructions=instructions,
            input_items=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analiza esta foto que el usuario acaba de mandar en mitad de una llamada.\n"
                                f"{context_text}"
                            ),
                        },
                        {"type": "input_image", "image_url": data.image_data_url},
                    ],
                }
            ],
            max_output_tokens=350,
        )

        text = self._extract_response_text(payload).strip()
        if not text:
            raise OpenAIClientError("No se pudo extraer una respuesta útil para la foto")

        if session.user_id is not None:
            billing_service.record_usage(
                user_id=session.user_id,
                session_id=data.session_id,
                provider="openai",
                endpoint="responses",
                model=self.openai.chat_model(),
                response_id=payload.get("id", ""),
                usage=payload.get("usage", {}) or {},
                metadata={"source": "realtime_photo_insight"},
            )
        session_service.append_memory(data.session_id, "assistant", text)
        return RealtimePhotoInsightResponse(text=text)

    def _extract_response_text(self, payload: dict) -> str:
        output = payload.get("output") or []
        chunks: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(str(content["text"]))
        return "\n".join(part.strip() for part in chunks if part and str(part).strip())


realtime_service = RealtimeService()
