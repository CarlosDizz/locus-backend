from __future__ import annotations

import json

from app.clients.openai_client import OpenAIClient, OpenAIClientError
from app.schemas.chat import ChatMessageRequest, ChatResponse, ChatSetupRequest
from app.schemas.poi import POI
from app.schemas.session import SessionCreateRequest, SessionUpdateRequest
from app.services.billing_service import BillingError, billing_service
from app.services.poi_service import poi_service
from app.services.prompt_service import prompt_service
from app.services.session_service import session_service
from app.services.tool_runtime_service import tool_runtime_service
from app.tools.knowledge_tools import get_knowledge_tool_manifest
from app.tools.poi_tools import get_poi_tool_manifest
from app.tools.session_tools import get_session_tool_manifest
from app.utils.text import clean_text


class ChatService:
    def __init__(self) -> None:
        self.openai = OpenAIClient()

    def _format_nearby_pois(self, pois: list[POI], limit: int = 8) -> str:
        if not pois:
            return ""
        lines: list[str] = []
        for poi in pois[:limit]:
            blurb = clean_text(poi.description or poi.summary)
            if blurb:
                lines.append(f"- {poi.name}: {blurb}")
            else:
                lines.append(f"- {poi.name}")
        return "\n".join(lines)

    def _format_ephemeral_map_pois(self, pois: list[POI], limit: int = 6) -> str:
        if not pois:
            return ""
        lines: list[str] = []
        for poi in pois[:limit]:
            blurb = clean_text(poi.description or poi.summary)
            if blurb:
                lines.append(f"- {poi.name}: ya marcado en el mapa de Locus como recomendacion temporal. {blurb}")
            else:
                lines.append(f"- {poi.name}: ya marcado en el mapa de Locus como recomendacion temporal.")
        return "\n".join(lines)

    def _format_location_context(self, session) -> str:
        if session.ephemeral_map_pois:
            ephemeral_names = ", ".join(poi.name for poi in session.ephemeral_map_pois[:3])
            if session.nearby_pois:
                visible_names = ", ".join(poi.name for poi in session.nearby_pois[:4])
                return (
                    "Hay geolocalizacion activa y el usuario esta en una zona donde el mapa ya muestra "
                    f"lugares como {visible_names}. Ademas, ya hay recomendaciones temporales marcadas en el mapa de Locus como {ephemeral_names}. "
                    "Usa ese contexto y no menciones coordenadas."
                )
            return (
                "Hay geolocalizacion activa y ya hay recomendaciones temporales marcadas en el mapa de Locus como "
                f"{ephemeral_names}. Usa ese contexto y no menciones coordenadas."
            )

        if session.nearby_pois:
            names = ", ".join(poi.name for poi in session.nearby_pois[:4])
            return (
                "Hay geolocalizacion activa y el usuario esta en una zona donde el mapa ya muestra "
                f"lugares como {names}. Usa ese contexto y no menciones coordenadas."
            )

        if session.location.lat is not None and session.location.lng is not None:
            return "Hay geolocalizacion activa de la sesion. Responde con cercania real, pero no menciones coordenadas."

        return ""

    def _format_active_poi_context(self, session) -> str:
        if session.active_poi is None:
            return ""
        blurb = clean_text(session.active_poi.description or session.active_poi.summary)
        if blurb:
            return f"{session.active_poi.name}. {blurb}"
        return session.active_poi.name

    def _format_recent_memory(self, session) -> str:
        return "\n".join(f"{item['role'].upper()}: {item['text']}" for item in session.memory[-8:])

    def _build_instructions(self, session_id: str) -> str:
        session = session_service.get_or_create(session_id)
        return prompt_service.render(
            "chat_agent.json",
            {
                "session_profile": session.profile.raw_context,
                "active_poi": self._format_active_poi_context(session),
                "session_location": self._format_location_context(session),
                "nearby_pois": self._format_nearby_pois(session.nearby_pois),
                "ephemeral_map_pois": self._format_ephemeral_map_pois(session.ephemeral_map_pois),
                "recent_memory": self._format_recent_memory(session),
            },
        )

    def _ensure_session_map_context(self, session_id: str):
        session = session_service.get_or_create(session_id)
        if session.nearby_pois:
            return session
        if session.location.lat is None or session.location.lng is None:
            return session
        pois = poi_service.search_nearby_pois(
            query="lugares turisticos",
            lat=session.location.lat,
            lng=session.location.lng,
            limit=8,
        )
        return session_service.set_nearby_pois(session_id, pois)

    def _tool_manifest(self) -> list[dict]:
        return [
            *get_session_tool_manifest(),
            *get_poi_tool_manifest(),
            *get_knowledge_tool_manifest(),
        ]

    def _extract_response_text(self, response: dict) -> str:
        output_text = response.get("output_text")
        if output_text:
            return clean_text(output_text)

        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                content_type = content.get("type", "")
                if content_type in {"output_text", "text"}:
                    text_value = content.get("text", "")
                    if text_value:
                        texts.append(text_value)
        return clean_text(" ".join(texts))

    def _extract_function_calls(self, response: dict) -> list[dict]:
        return [item for item in response.get("output", []) if item.get("type") == "function_call"]

    def _run_openai_chat(self, session_id: str, user_message: str) -> tuple[str, dict]:
        self._ensure_session_map_context(session_id)
        instructions = self._build_instructions(session_id)
        session = session_service.get_or_create(session_id)

        response = self.openai.create_response(
            model=self.openai.chat_model(),
            instructions=instructions,
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": user_message}]}],
            tools=self._tool_manifest(),
            previous_response_id=session.metadata.get("last_chat_response_id"),
            max_output_tokens=300,
        )

        while True:
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                session_service.set_metadata_value(session_id, "last_chat_response_id", response.get("id", ""))
                return self._extract_response_text(response), response

            tool_outputs = []
            for call in function_calls:
                try:
                    arguments = json.loads(call.get("arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                output = tool_runtime_service.execute(session_id, call.get("name", ""), arguments)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": output,
                    }
                )

            response = self.openai.create_response(
                model=self.openai.chat_model(),
                instructions=instructions,
                input_items=tool_outputs,
                tools=self._tool_manifest(),
                previous_response_id=response.get("id"),
            )

    def setup_chat(self, data: ChatSetupRequest) -> ChatResponse:
        session = session_service.create_session(
            SessionCreateRequest(
                session_id=data.session_id,
                user_id=data.user_id,
                profile_context=data.profile_context,
                lat=data.lat,
                lng=data.lng,
            )
        )

        pois = poi_service.search_nearby_pois("lugares turisticos", data.lat, data.lng, limit=8)
        session = session_service.set_nearby_pois(session.session_id, pois)
        session = session_service.set_ephemeral_map_pois(session.session_id, [])
        prompt_preview = prompt_service.render(
            "chat_agent.json",
            {
                "session_profile": session.profile.raw_context,
                "active_poi": self._format_active_poi_context(session),
                "session_location": self._format_location_context(session),
                "nearby_pois": self._format_nearby_pois(session.nearby_pois),
                "ephemeral_map_pois": "",
                "recent_memory": "",
            },
        )

        reply = "Ya tengo vuestro perfil y puedo orientaros con lo que aparece en el mapa."
        if pois:
            highlighted = ", ".join(poi.name for poi in pois[:3])
            reply += f" Para empezar, me fijaria en {highlighted}."
        else:
            reply += " En cuanto tenga lugares claros cerca, te los propongo sin rodeos."

        return ChatResponse(
            session_id=session.session_id,
            reply=reply,
            pois=pois,
            prompt_preview=prompt_preview,
        )

    def handle_message(self, data: ChatMessageRequest) -> ChatResponse:
        if data.user_id is not None:
            session_service.attach_user(data.session_id, data.user_id)
        session = session_service.update_session(
            data.session_id,
            SessionUpdateRequest(user_id=data.user_id, lat=data.lat, lng=data.lng),
        )
        session = self._ensure_session_map_context(session.session_id)

        clean_message = clean_text(data.message)
        session_service.append_memory(session.session_id, "user", clean_message)
        try:
            if self.openai.is_configured():
                if session.user_id is not None:
                    billing_service.ensure_user_can_consume(session.user_id)
                reply, final_response = self._run_openai_chat(session.session_id, clean_message)
                final_session = session_service.get_or_create(session.session_id)
                if final_session.user_id is not None:
                    billing_service.record_usage(
                        user_id=final_session.user_id,
                        session_id=final_session.session_id,
                        provider="openai",
                        endpoint="responses",
                        model=self.openai.chat_model(),
                        response_id=final_response.get("id", ""),
                        usage=final_response.get("usage", {}) or {},
                        metadata={"source": "chat_service"},
                    )
            else:
                raise OpenAIClientError("OpenAI no configurado")
        except BillingError:
            reply = "Tu saldo actual no permite iniciar una nueva interacción. Recarga saldo para seguir usando Locus."
        except OpenAIClientError:
            reply = "Ahora mismo no he podido responder bien. Inténtalo otra vez en un momento."

        latest_session = session_service.get_or_create(session.session_id)
        pois = latest_session.nearby_pois
        ephemeral_pois = list(latest_session.ephemeral_map_pois)
        prompt_preview = self._build_instructions(session.session_id)
        session_service.append_memory(session.session_id, "assistant", reply)
        return ChatResponse(
            session_id=session.session_id,
            reply=reply,
            pois=pois,
            ephemeral_pois=ephemeral_pois,
            prompt_preview=prompt_preview,
        )


chat_service = ChatService()
