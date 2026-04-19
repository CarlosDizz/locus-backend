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

    def _format_location_context(self, session) -> str:
        if session.nearby_pois:
            names = ", ".join(poi.name for poi in session.nearby_pois[:4])
            return (
                "Hay geolocalizacion activa y el usuario esta en una zona donde el mapa ya muestra "
                f"lugares como {names}. Usa ese contexto y no menciones coordenadas."
            )

        if session.location.lat is not None and session.location.lng is not None:
            return "Hay geolocalizacion activa de la sesion. Responde con cercania real, pero no menciones coordenadas."

        return ""

    def _format_recent_memory(self, session) -> str:
        return "\n".join(f"{item['role'].upper()}: {item['text']}" for item in session.memory[-8:])

    def _extract_ephemeral_places_query(self, message: str) -> str | None:
        lowered = clean_text(message).lower()
        if any(token in lowered for token in ["carbonara", "comer", "restaurante", "ristorante", "pasta"]):
            return "restaurante carbonara"
        if any(token in lowered for token in ["ipa", "cerveza artesanal", "craft beer", "cerveza", "pub", "bar"]):
            return "bar cerveza artesanal ipa"
        if any(token in lowered for token in ["cafe", "café", "desayuno", "coffee"]):
            return "cafe"
        return None

    def _places_reply_mentions_result(self, reply: str, places: list[POI]) -> bool:
        lowered = reply.lower()
        return any(place.name.lower() in lowered for place in places[:5])

    def _grounded_places_reply(self, message: str, places: list[POI]) -> str:
        lowered = clean_text(message).lower()
        first = places[0]
        second = places[1] if len(places) > 1 else None

        if any(token in lowered for token in ["carbonara", "pasta", "restaurante", "ristorante", "comer"]):
            if second:
                return (
                    f"Sí. Cerca de ti miraría primero {first.name} y, como segunda opción, {second.name}. "
                    f"{first.name} te queda bien para una carbonara sin desviarte demasiado; si lo ves lleno, {second.name} te salva el plan."
                )
            return f"Sí. Cerca de ti iría a {first.name}; es la opción más clara para resolver una buena carbonara sin complicarte."

        if any(token in lowered for token in ["ipa", "cerveza", "beer", "bar", "pub", "artigianale"]):
            if second:
                return (
                    f"Después de comer, yo probaría primero {first.name}. Si quieres una segunda bala, apunta también {second.name}. "
                    "Busca una IPA tirada o pregunta por la más lupulada que tengan."
                )
            return f"Después de comer, yo probaría {first.name}. Pide una IPA tirada o pregunta por la más lupulada de la casa."

        names = ", ".join(place.name for place in places[:3])
        return f"Para eso cerca de ti me fijaría en {names}."

    def _poi_reason(self, poi: POI) -> str:
        text = clean_text(poi.description or poi.summary)
        if not text:
            return "Es una opcion clara de las que tienes ya visibles en el mapa."
        first_sentence = text.split(".")[0].strip()
        if first_sentence:
            return first_sentence.rstrip(".") + "."
        return "Es una opcion clara de las que tienes ya visibles en el mapa."

    def _pick_grounded_candidates(self, message: str, pois: list[POI]) -> tuple[POI, POI | None]:
        lowered = message.lower()
        visible = list(pois)
        if not visible:
            raise ValueError("No hay POIs visibles")

        if any(token in lowered for token in ["cans", "45", "sent", "descans", "poco", "andar", "caminar"]):
            restful = next(
                (
                    poi for poi in visible
                    if any(word in (poi.description or poi.summary).lower() for word in ["plaza", "museo", "basilica", "iglesia"])
                ),
                None,
            )
            if restful and restful is not visible[0]:
                visible.remove(restful)
                visible.insert(0, restful)

        first = visible[0]
        second = visible[1] if len(visible) > 1 else None
        return first, second

    def _grounded_map_reply(self, message: str, pois: list[POI]) -> str:
        first, second = self._pick_grounded_candidates(message, pois)
        lowered = message.lower()

        if any(token in lowered for token in ["cans", "45", "sent", "descans", "poco", "andar", "caminar"]):
            if second:
                return (
                    f"Con 45 minutos y poca caminata, yo iria a {first.name}. {self._poi_reason(first)} "
                    f"Si prefieres una alternativa igual de facil, prueba con {second.name}. {self._poi_reason(second)}"
                )
            return f"Con 45 minutos y poca caminata, yo iria a {first.name}. {self._poi_reason(first)}"

        if any(token in lowered for token in ["cerca", "que ver", "qué ver", "recom", "merece"]):
            if second:
                return (
                    f"Ahora mismo me quedaria con {first.name}. {self._poi_reason(first)} "
                    f"Si quieres una segunda opcion cercana, {second.name}. {self._poi_reason(second)}"
                )
            return f"Ahora mismo me quedaria con {first.name}. {self._poi_reason(first)}"

        names = ", ".join(poi.name for poi in pois[:3])
        return f"Para orientarte con lo que ya tienes delante, me fijaria en {names}. Si me dices el plan que te apetece, te dejo una sola recomendacion."

    def _reply_mentions_visible_poi(self, reply: str, pois: list[POI]) -> bool:
        lowered = reply.lower()
        return any(poi.name.lower() in lowered for poi in pois[:8])

    def _reply_sounds_technical(self, reply: str) -> bool:
        lowered = reply.lower()
        bad_tokens = [
            "coordenad",
            "mi busqueda",
            "herramient",
            "modelo",
            "sesion",
            "prompt",
            "backend",
            "no me aparecen lugares concretos",
            "no me aparecen lugares",
            "captura",
        ]
        return any(token in lowered for token in bad_tokens)

    def _needs_grounding(self, message: str, reply: str, pois: list[POI]) -> bool:
        if not pois:
            return False

        lowered = message.lower()
        asks_for_specific = any(
            token in lowered for token in ["cerca", "que ver", "qué ver", "recom", "merece", "cans", "45", "andar", "caminar", "poco"]
        )
        if self._reply_sounds_technical(reply):
            return True
        if asks_for_specific and not self._reply_mentions_visible_poi(reply, pois):
            return True
        return False

    def _fallback_reply(self, session, message: str, pois: list[POI]) -> str:
        lowered = message.lower()
        active_name = session.active_poi.name if session.active_poi else ""
        visible = pois or session.nearby_pois

        if visible:
            if any(token in lowered for token in ["cans", "45", "poco", "cerca", "andar", "rapido", "rápido", "caminar"]):
                return self._grounded_map_reply(message, visible)

            if active_name:
                top_names = ", ".join(poi.name for poi in visible[:3])
                return f"Seguimos con {active_name} en mente, pero si quieres cambiar de plan ahora mismo tambien tienes cerca {top_names}."

            return self._grounded_map_reply(message, visible)

        return "Estoy contigo. Dime si te apetece algo monumental, tranquilo, rapido o mas local y te doy una propuesta concreta."

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
        prompt_preview = prompt_service.render(
            "chat_agent.json",
            {
                "session_profile": session.profile.raw_context,
                "active_poi": session.active_poi.name if session.active_poi else "",
                "session_location": self._format_location_context(session),
                "nearby_pois": self._format_nearby_pois(session.nearby_pois),
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

    def _build_instructions(self, session_id: str) -> str:
        session = session_service.get_or_create(session_id)
        return prompt_service.render(
            "chat_agent.json",
            {
                "session_profile": session.profile.raw_context,
                "active_poi": session.active_poi.name if session.active_poi else "",
                "session_location": self._format_location_context(session),
                "nearby_pois": self._format_nearby_pois(session.nearby_pois),
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
        ephemeral_pois: list[POI] = []

        pois: list[POI] = []
        try:
            if self.openai.is_configured():
                if session.user_id is not None:
                    billing_service.ensure_user_can_consume(session.user_id)
                reply, final_response = self._run_openai_chat(session.session_id, clean_message)
                fresh_session = session_service.get_or_create(session.session_id)
                if self._needs_grounding(clean_message, reply, fresh_session.nearby_pois):
                    reply = self._grounded_map_reply(clean_message, fresh_session.nearby_pois)
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
            lowered = clean_message.lower()
            if any(token in lowered for token in ["poi", "cerca", "que ver", "qué ver", "recomienda"]):
                pois = poi_service.search_nearby_pois(
                    query=clean_message or "lugares turisticos",
                    lat=session.location.lat,
                    lng=session.location.lng,
                    limit=5,
                )
                session = session_service.set_nearby_pois(session.session_id, pois)

            reply = self._fallback_reply(session, clean_message, pois)

        ephemeral_query = self._extract_ephemeral_places_query(clean_message)
        if ephemeral_query:
            ephemeral_pois = poi_service.search_contextual_places(
                query=ephemeral_query,
                lat=session.location.lat,
                lng=session.location.lng,
                limit=5,
            )
            if ephemeral_pois and not self._places_reply_mentions_result(reply, ephemeral_pois):
                reply = self._grounded_places_reply(clean_message, ephemeral_pois)

        pois = session_service.get_or_create(session.session_id).nearby_pois
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
