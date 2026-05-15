from __future__ import annotations

import json
from time import perf_counter

from app.clients.openai_client import OpenAIClient, OpenAIClientError
from app.config import settings
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
from app.tools.referral_tools import get_referral_tool_manifest
from app.tools.session_tools import get_session_tool_manifest
from app.utils.logging import get_logger
from app.utils.text import clean_text, slugify


class ChatService:
    def __init__(self) -> None:
        self.openai = OpenAIClient()
        self.logger = get_logger(__name__)

    def _message_suggests_missing_landmark(self, message: str) -> bool:
        lowered = clean_text(message).lower()
        if not lowered:
            return False
        markers = [
            "no lo encuentro",
            "no la encuentro",
            "no aparece",
            "no esta en el mapa",
            "no está en el mapa",
            "falta en el mapa",
            "falta este sitio",
            "falta este lugar",
            "deberia estar",
            "debería estar",
            "me extraña",
            "entre las sugerencias",
            "entre los puntos de interes",
            "entre los puntos de interés",
            "en el mapa",
            "punto de interes",
            "punto de interés",
        ]
        return any(marker in lowered for marker in markers)

    def _message_suggests_visit_intent(self, message: str) -> bool:
        lowered = clean_text(message).lower()
        markers = [
            "quiero visitar",
            "me gustaria visitar",
            "me gustaría visitar",
            "me gustaria verlo",
            "me gustaría verlo",
            "quiero verlo",
            "verlo por fuera",
            "visitarlo",
            "hacer una visita",
            "hacer visita",
            "se puede visitar",
            "puedo visitarlo",
            "puedo visitarla",
            "merece la pena",
            "vale la pena",
            "voy a ir",
            "ir al",
            "ir a la",
            "turistico",
            "turístico",
            "cultural",
        ]
        return any(marker in lowered for marker in markers)

    def _message_suggests_web_research(self, message: str) -> bool:
        lowered = clean_text(message).lower()
        if not lowered:
            return False
        markers = [
            "quien era",
            "quién era",
            "quien fue",
            "quién fue",
            "historia de",
            "entrada",
            "entradas",
            "tarifa",
            "tarifas",
            "precio",
            "precios",
            "comprar",
            "reservar",
            "reserva",
            "horario",
            "horarios",
            "gratis",
            "gratuito",
            "curiosidad",
            "curiosidades",
            "por que se llama",
            "por qué se llama",
            "origen del nombre",
            "investiga",
            "busca informacion",
            "busca información",
            "cuentame mas",
            "cuéntame más",
            "que sabes de",
            "qué sabes de",
        ]
        return any(marker in lowered for marker in markers)

    def _select_catalog_promotion_candidate(self, session, user_message: str) -> POI | None:
        message_slug = slugify(user_message)
        message_tokens = {token for token in message_slug.split("-") if len(token) > 2}

        candidates: list[POI] = []
        if session.active_poi is not None:
            candidates.append(session.active_poi)
        candidates.extend(session.ephemeral_map_pois)

        raw_tool_candidates = session.metadata.get(tool_runtime_service.TOOL_CANDIDATES_METADATA_KEY) or []
        for item in raw_tool_candidates:
            try:
                candidates.append(POI(**item))
            except Exception:
                continue

        seen: set[str] = set()
        deduped: list[POI] = []
        for poi in candidates:
            key = slugify(poi.name)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(poi)

        if not deduped:
            return None

        if len(deduped) == 1 and (self._message_suggests_missing_landmark(user_message) or self._message_suggests_visit_intent(user_message)):
            return deduped[0]

        ranked: list[tuple[int, POI]] = []
        for poi in deduped:
            poi_tokens = {token for token in slugify(poi.name).split("-") if len(token) > 2}
            overlap = len(message_tokens & poi_tokens)
            score = overlap * 10
            if poi == session.active_poi:
                score += 4
            if poi.is_ephemeral:
                score += 2
            cultural_tokens = {
                "teatro", "museo", "catedral", "circo", "interpretacion", "interpretación",
                "visitantes", "mirador", "patrimonio", "agua", "depositos", "depósitos",
                "centro", "cultural", "arqueologico", "arqueológico",
            }
            if poi_tokens & cultural_tokens:
                score += 2
            ranked.append((score, poi))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1] if ranked and ranked[0][0] > 0 else None

    def _maybe_promote_candidate_to_catalog(self, session_id: str, user_message: str) -> dict | None:
        session = session_service.get_or_create(session_id)
        should_try = self._message_suggests_missing_landmark(user_message) or self._message_suggests_visit_intent(user_message)
        if not should_try:
            return None

        candidate = self._select_catalog_promotion_candidate(session, user_message)
        if candidate is None:
            return None

        try:
            result = json.loads(
                tool_runtime_service.execute(
                    session_id,
                    "promote_poi_to_catalog",
                    {
                        "poi_name": candidate.name,
                        "reason": "El usuario lo echa en falta en el mapa o quiere visitarlo como punto importante de la ciudad.",
                    },
                )
            )
        except Exception:
            return None

        if not result.get("ok"):
            return result
        return result

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

    def _format_session_profile(self, session) -> str:
        preferences = session.profile.preferences or {}
        preferred_name = clean_text(str(preferences.get("preferred_name") or ""))
        raw_context = clean_text(session.profile.raw_context or "")
        parts: list[str] = []
        if preferred_name:
            parts.append(f"Quiere que le llames {preferred_name}.")
        if raw_context:
            parts.append(raw_context)
        return " ".join(parts).strip() or "No hay preferencias personales guardadas todavía."

    def _build_instructions(self, session_id: str, session=None) -> str:
        session = session or session_service.get_or_create(session_id)
        return prompt_service.render(
            "chat_agent.json",
            {
                "session_profile": self._format_session_profile(session),
                "active_poi": self._format_active_poi_context(session),
                "session_location": self._format_location_context(session),
                "nearby_pois": self._format_nearby_pois(session.nearby_pois),
                "ephemeral_map_pois": self._format_ephemeral_map_pois(session.ephemeral_map_pois),
                "recent_memory": self._format_recent_memory(session),
            },
        )

    def _poi_centroid_distance_km(self, lat: float, lng: float, pois: list) -> float:
        avg_lat = sum(p.lat for p in pois) / len(pois)
        avg_lng = sum(p.lng for p in pois) / len(pois)
        from math import cos, radians, sqrt
        lat_km = (lat - avg_lat) * 111.32
        lng_km = (lng - avg_lng) * 111.32 * max(cos(radians(lat)), 0.2)
        return sqrt(lat_km ** 2 + lng_km ** 2)

    def _ensure_session_map_context(self, session_id: str):
        session = session_service.get_or_create(session_id)
        if session.location.lat is None or session.location.lng is None:
            return session
        if session.nearby_pois:
            dist = self._poi_centroid_distance_km(session.location.lat, session.location.lng, session.nearby_pois)
            if dist < 10.0:
                return session
        pois = poi_service.search_nearby_pois(
            query="lugares turisticos",
            lat=session.location.lat,
            lng=session.location.lng,
            limit=8,
        )
        return session_service.set_nearby_pois(session_id, pois)

    def _tool_manifest(self, *, include_web_search: bool = False) -> list[dict]:
        tools: list[dict] = [
            *get_session_tool_manifest(),
            *get_poi_tool_manifest(),
            *get_knowledge_tool_manifest(include_web_research_tool=False),
            *get_referral_tool_manifest(),
        ]
        if include_web_search:
            tools.append(
                {
                    "type": "web_search",
                    "user_location": {
                        "type": "approximate",
                        "country": "ES",
                        "timezone": "Europe/Madrid",
                    },
                }
            )
        return tools

    def _create_chat_response(
        self,
        *,
        instructions: str,
        input_items: list[dict],
        previous_response_id: str | None,
        include_web_search: bool = False,
        max_output_tokens: int = 700,
    ) -> dict:
        full_tools = self._tool_manifest(include_web_search=include_web_search)
        no_web_tools = [tool for tool in full_tools if tool.get("type") != "web_search"]
        attempts = [
            ("full", full_tools, previous_response_id),
            ("no_web_search", no_web_tools, previous_response_id),
            ("fresh_context", no_web_tools, None),
            ("no_tools", None, None),
        ]
        last_error: OpenAIClientError | None = None

        for label, tools, prev_id in attempts:
            if label == "no_web_search" and tools == full_tools:
                continue
            try:
                if label != "full":
                    self.logger.warning(
                        "chat_turn retry strategy=%s previous_response_id=%s model=%s",
                        label,
                        prev_id or "",
                        self.openai.chat_model(),
                    )
                return self.openai.create_response(
                    model=self.openai.chat_model(),
                    instructions=instructions,
                    input_items=input_items,
                    tools=tools,
                    previous_response_id=prev_id,
                    max_output_tokens=max_output_tokens,
                )
            except OpenAIClientError as exc:
                last_error = exc
                self.logger.warning(
                    "chat_turn openai_attempt_failed strategy=%s previous_response_id=%s model=%s error=%s",
                    label,
                    prev_id or "",
                    self.openai.chat_model(),
                    str(exc),
                )
                continue

        if last_error is not None:
            raise last_error
        raise OpenAIClientError("Responses API failed without a captured error")

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

    def _run_openai_chat(self, session_id: str, user_message: str) -> tuple[str, dict, dict]:
        flow_started_at = perf_counter()
        context_started_at = perf_counter()
        session = self._ensure_session_map_context(session_id)
        instructions = self._build_instructions(session_id, session)
        metrics = {
            "context_ms": round((perf_counter() - context_started_at) * 1000, 1),
            "openai_ms": 0.0,
            "tools_ms": 0.0,
            "openai_rounds": 0,
            "tool_calls": 0,
            "tool_batches": 0,
        }
        include_web_search = settings.openai_chat_enable_web_search and self._message_suggests_web_research(user_message)

        openai_started_at = perf_counter()
        response = self._create_chat_response(
            instructions=instructions,
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": user_message}]}],
            previous_response_id=session.metadata.get("last_chat_response_id"),
            include_web_search=include_web_search,
            max_output_tokens=300,
        )
        metrics["openai_ms"] += round((perf_counter() - openai_started_at) * 1000, 1)
        metrics["openai_rounds"] += 1

        while True:
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                session_service.set_metadata_value(session_id, "last_chat_response_id", response.get("id", ""))
                metrics["flow_ms"] = round((perf_counter() - flow_started_at) * 1000, 1)
                return self._extract_response_text(response), response, metrics

            metrics["tool_batches"] += 1
            metrics["tool_calls"] += len(function_calls)
            tool_outputs = []
            tools_started_at = perf_counter()
            for call in function_calls:
                try:
                    arguments = json.loads(call.get("arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_name = call.get("name", "")
                self.logger.warning(
                    "chat_tool_call session=%s tool=%s arguments=%s",
                    session_id,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False)[:1000],
                )
                print(
                    "chat_tool_call "
                    f"session={session_id} "
                    f"tool={tool_name} "
                    f"arguments={json.dumps(arguments, ensure_ascii=False)[:1000]}",
                    flush=True,
                )
                output = tool_runtime_service.execute(session_id, call.get("name", ""), arguments)
                self.logger.warning(
                    "chat_tool_output session=%s tool=%s output=%s",
                    session_id,
                    tool_name,
                    output[:1200],
                )
                print(
                    "chat_tool_output "
                    f"session={session_id} "
                    f"tool={tool_name} "
                    f"output={output[:1200]}",
                    flush=True,
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": output,
                    }
                )
            metrics["tools_ms"] += round((perf_counter() - tools_started_at) * 1000, 1)

            openai_started_at = perf_counter()
            response = self._create_chat_response(
                instructions=instructions,
                input_items=tool_outputs,
                previous_response_id=response.get("id"),
                include_web_search=include_web_search,
            )
            metrics["openai_ms"] += round((perf_counter() - openai_started_at) * 1000, 1)
            metrics["openai_rounds"] += 1

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
                "session_profile": self._format_session_profile(session),
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
        turn_started_at = perf_counter()
        session_started_at = perf_counter()
        if data.user_id is not None:
            session_service.attach_user(data.session_id, data.user_id)
        session = session_service.update_session(
            data.session_id,
            SessionUpdateRequest(user_id=data.user_id, lat=data.lat, lng=data.lng),
        )
        session_ms = round((perf_counter() - session_started_at) * 1000, 1)

        clean_message = clean_text(data.message)
        session_service.append_memory(session.session_id, "user", clean_message)
        chat_metrics = {
            "context_ms": 0.0,
            "openai_ms": 0.0,
            "tools_ms": 0.0,
            "openai_rounds": 0,
            "tool_calls": 0,
            "tool_batches": 0,
            "flow_ms": 0.0,
        }
        promotion_ms = 0.0
        outcome = "ok"
        try:
            if self.openai.is_configured():
                if session.user_id is not None:
                    billing_service.ensure_user_can_consume(session.user_id)
                reply, final_response, chat_metrics = self._run_openai_chat(session.session_id, clean_message)
                promotion_started_at = perf_counter()
                promotion_result = self._maybe_promote_candidate_to_catalog(session.session_id, clean_message)
                promotion_ms = round((perf_counter() - promotion_started_at) * 1000, 1)
                if promotion_result and promotion_result.get("ok"):
                    promoted_name = clean_text(str(promotion_result.get("poi_name", "")))
                    if promoted_name:
                        reply = (
                            f"{reply}\n\n"
                            f"Además, ya te lo he incorporado también como punto de interés normal del mapa para que puedas abrir su ficha y usarlo como visita."
                        )
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
                        metadata={
                            "source": "chat_service",
                            "interaction_type": "chat",
                        },
                    )
            else:
                raise OpenAIClientError("OpenAI no configurado")
        except BillingError:
            outcome = "billing_blocked"
            reply = "Tu saldo actual no permite iniciar una nueva interacción. Recarga saldo para seguir usando Locus."
        except OpenAIClientError:
            outcome = "openai_error"
            reply = "Ahora mismo no he podido responder bien. Inténtalo otra vez en un momento."

        finalize_started_at = perf_counter()
        latest_session = session_service.append_memory(session.session_id, "assistant", reply)
        pois = latest_session.nearby_pois
        ephemeral_pois = list(latest_session.ephemeral_map_pois)
        finalize_ms = round((perf_counter() - finalize_started_at) * 1000, 1)
        total_ms = round((perf_counter() - turn_started_at) * 1000, 1)
        timing_line = (
            "chat_turn "
            f"session={session.session_id} "
            f"outcome={outcome} "
            f"total_ms={total_ms:.1f} "
            f"session_ms={session_ms:.1f} "
            f"context_ms={chat_metrics['context_ms']:.1f} "
            f"openai_ms={chat_metrics['openai_ms']:.1f} "
            f"rounds={chat_metrics['openai_rounds']} "
            f"tool_batches={chat_metrics['tool_batches']} "
            f"tool_calls={chat_metrics['tool_calls']} "
            f"tools_ms={chat_metrics['tools_ms']:.1f} "
            f"promotion_ms={promotion_ms:.1f} "
            f"finalize_ms={finalize_ms:.1f}"
        )
        self.logger.warning(timing_line)
        print(timing_line, flush=True)
        return ChatResponse(
            session_id=session.session_id,
            reply=reply,
            pois=pois,
            ephemeral_pois=ephemeral_pois,
            prompt_preview="",
        )


chat_service = ChatService()
