from app.clients.openai_client import OpenAIClient, OpenAIClientError
from app.schemas.realtime import RealtimeClientSecretResponse, RealtimeSessionRequest, RealtimeSessionResponse
from app.services.billing_service import billing_service
from app.services.prompt_service import prompt_service
from app.services.session_service import session_service
from app.tools.knowledge_tools import get_knowledge_tool_manifest
from app.tools.poi_tools import get_poi_tool_manifest
from app.tools.session_tools import get_session_tool_manifest


class RealtimeService:
    def __init__(self) -> None:
        self.openai = OpenAIClient()

    def prepare_session(self, data: RealtimeSessionRequest) -> RealtimeSessionResponse:
        if data.user_id is not None:
            session_service.attach_user(data.session_id, data.user_id)
        session = session_service.get_or_create(data.session_id)
        active_poi_name = data.active_poi_name or (session.active_poi.name if session.active_poi else "")
        instructions = prompt_service.render(
            "realtime_agent.json",
            {
                "session_profile": session.profile.raw_context,
                "active_poi": active_poi_name,
                "visit_context": data.visit_context,
                "recent_memory": "\n".join(
                    f"{item['role'].upper()}: {item['text']}" for item in session.memory[-8:]
                ),
            },
        )
        return RealtimeSessionResponse(
            session_id=session.session_id,
            model=self.openai.realtime_model(),
            transport="webrtc",
            webrtc_api_url="https://api.openai.com/v1/realtime",
            instructions=instructions,
            modalities=["text", "audio", "image"],
            tools=[
                *get_session_tool_manifest(),
                *get_poi_tool_manifest(),
                *get_knowledge_tool_manifest(),
            ],
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
            max_output_tokens=500,
        )
        client_secret = payload.get("client_secret", {}) or payload.get("session", {}).get("client_secret", {})
        expires_at = client_secret.get("expires_at") or payload.get("expires_at") or 0
        value = client_secret.get("value") or payload.get("value") or ""
        return RealtimeClientSecretResponse(
            session_id=prepared.session_id,
            model=prepared.model,
            client_secret=value,
            expires_at=expires_at,
            webrtc_api_url="https://api.openai.com/v1/realtime",
            instructions=prepared.instructions,
            voice=self.openai.realtime_voice(),
            modalities=["audio"],
            tools=prepared.tools,
        )


realtime_service = RealtimeService()
