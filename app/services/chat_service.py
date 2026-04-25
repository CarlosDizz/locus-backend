from __future__ import annotations

import json
import re

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
from app.utils.text import clean_text, slugify


class ChatService:
    MATCH_STOPWORDS = {
        "de",
        "del",
        "la",
        "las",
        "el",
        "los",
        "y",
        "of",
        "the",
        "di",
        "da",
        "do",
        "a",
        "al",
        "en",
        "un",
        "una",
        "por",
    }

    LANDMARK_TERMS = [
        "obelisco",
        "estatua",
        "monumento",
        "fuente",
        "arco",
        "torre",
        "puerta",
        "muralla",
        "castillo",
        "puente",
        "plaza",
        "edificio",
        "iglesia",
        "basilica",
        "basílica",
        "palacio",
        "templo",
        "teatro",
        "mirador",
    ]

    RELATIVE_LOCATION_TERMS = [
        "frente",
        "delante",
        "enfrente",
        "junto",
        "al lado",
        "cerca",
        "pegado",
        "justo",
    ]

    DEICTIC_LANDMARK_TERMS = [
        "este",
        "esta",
        "esto",
        "ese",
        "esa",
        "eso",
        "aqui",
        "aquí",
        "veo",
        "viendo",
        "tengo delante",
        "puerta principal",
        "te estoy comentando",
    ]

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

    def _normalize_match_text(self, text: str) -> str:
        return slugify(clean_text(text)).replace("-", " ")

    def _match_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in self._normalize_match_text(text).split()
            if token and token not in self.MATCH_STOPWORDS and len(token) > 2
        }

    def _poi_match_score(self, message: str, poi: POI) -> int:
        normalized_message = self._normalize_match_text(message)
        normalized_name = self._normalize_match_text(poi.name)
        if normalized_name and normalized_name in normalized_message:
            return 100

        poi_tokens = self._match_tokens(poi.name)
        message_tokens = self._match_tokens(message)
        overlap = poi_tokens & message_tokens
        if len(overlap) >= 2:
            return len(overlap) * 10
        if poi_tokens and overlap == poi_tokens:
            return len(overlap) * 10
        return 0

    def _find_named_poi(self, message: str, pois: list[POI]) -> POI | None:
        best: POI | None = None
        best_score = 0
        ambiguous = False
        for poi in pois:
            score = self._poi_match_score(message, poi)
            if score > best_score:
                best = poi
                best_score = score
                ambiguous = False
            elif score > 0 and score == best_score:
                ambiguous = True
        if best_score <= 0 or ambiguous:
            return None
        return best

    def _find_named_focus_poi(self, message: str, session) -> POI | None:
        named_visible = self._find_named_poi(message, session.nearby_pois)
        if named_visible is not None:
            return named_visible
        return self._find_named_poi(message, session.ephemeral_map_pois)

    def _contains_term(self, text: str, term: str) -> bool:
        if not term:
            return False
        if " " in term:
            return term in text
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None

    def _contains_any_term(self, text: str, terms: list[str]) -> bool:
        return any(self._contains_term(text, term) for term in terms)

    def _looks_like_landmark_reference(self, message: str) -> bool:
        lowered = clean_text(message).lower()
        if not self._contains_any_term(lowered, self.LANDMARK_TERMS):
            return False
        return self._contains_any_term(lowered, self.RELATIVE_LOCATION_TERMS) or self._contains_any_term(lowered, self.DEICTIC_LANDMARK_TERMS)

    def _recent_user_landmark_reference(self, session) -> str:
        for item in reversed(session.memory[-6:]):
            if item.get("role") != "user":
                continue
            text = clean_text(item.get("text", ""))
            if text and self._looks_like_landmark_reference(text):
                return text
        return ""

    def _extract_landmark_reference_text(self, session, message: str) -> str:
        cleaned = clean_text(message)
        lowered = cleaned.lower()
        if self._looks_like_landmark_reference(cleaned):
            return cleaned

        if session.active_poi is None:
            return ""

        asks_about_this = self._contains_any_term(lowered, ["este", "esta", "esto", "ese", "esa", "eso", "te estoy comentando"])
        mentions_landmark = self._contains_any_term(lowered, self.LANDMARK_TERMS)
        if asks_about_this or mentions_landmark:
            previous = self._recent_user_landmark_reference(session)
            if previous:
                return previous
        return ""

    def _references_external_map(self, message: str) -> bool:
        lowered = clean_text(message).lower()
        return any(token in lowered for token in ["google maps", "apple maps", "waze", "maps de apple"])

    def _asks_about_map_presence(self, message: str) -> bool:
        lowered = clean_text(message).lower()
        return "mapa" in lowered and any(
            token in lowered
            for token in [
                "encontr",
                "aparece",
                "aparecera",
                "aparecerá",
                "marcad",
                "veré",
                "vere",
                "estará",
                "estara",
            ]
        )

    def _extract_ephemeral_places_query(self, message: str) -> str | None:
        lowered = clean_text(message).lower()
        if self._contains_any_term(lowered, ["carbonara", "comer", "restaurante", "ristorante", "pasta"]):
            return "restaurante carbonara"
        if self._contains_any_term(lowered, ["ipa", "cerveza artesanal", "craft beer", "cerveza", "pub", "bar"]):
            return "bar cerveza artesanal ipa"
        if self._contains_any_term(lowered, ["cafe", "café", "desayuno", "coffee"]):
            return "cafe"
        return None

    def _classify_request_intent(self, message: str) -> str:
        lowered = clean_text(message).lower()

        if self._contains_any_term(lowered, ["carbonara", "restaurante", "ristorante", "comer", "cenar", "tapeo", "pizza", "pasta"]):
            return "hospitality_food"
        if self._contains_any_term(lowered, ["ipa", "cerveza", "beer", "bar", "pub", "coctel", "cóctel", "vino", "birra"]):
            return "hospitality_drink"
        if self._contains_any_term(lowered, ["farmacia", "atm", "cajero", "taxi", "supermercado", "hospital", "urgencia", "lavanderia", "lavandería"]):
            return "service"
        if self._contains_any_term(
            lowered,
            [
                "quiero visitar",
                "quiero ver",
                "merece la pena",
                "merece más la pena",
                "que es",
                "qué es",
                "háblame de",
                "hablame de",
                "edificio historico",
                "edificio histórico",
                "museo",
                "iglesia",
                "basilica",
                "basílica",
                "fabrica",
                "fábrica",
                "castillo",
                "puente",
                "monumento",
            ],
        ):
            return "tourism_candidate"
        return "general"

    def _extract_tourism_candidate_query(self, message: str, active_poi_name: str = "") -> str:
        cleaned = clean_text(message)
        lowered = cleaned.lower()
        landmark_term = next((term for term in self.LANDMARK_TERMS if self._contains_term(lowered, term)), "")
        has_relative_reference = self._contains_any_term(lowered, self.RELATIVE_LOCATION_TERMS)
        if landmark_term and active_poi_name and has_relative_reference:
            return f"{landmark_term} {clean_text(active_poi_name)}"
        if landmark_term and active_poi_name and len(lowered.split()) <= 6:
            return f"{landmark_term} {clean_text(active_poi_name)}"
        if landmark_term:
            return landmark_term

        replacements = [
            "quiero visitar ",
            "quiero ver ",
            "háblame de ",
            "hablame de ",
            "qué es ",
            "que es ",
            "me interesa ",
            "quiero conocer ",
        ]
        query = cleaned
        for prefix in replacements:
            if lowered.startswith(prefix):
                query = cleaned[len(prefix):]
                break

        for suffix in [
            ", que es un edificio histórico",
            ", que es un edificio historico",
            " que es un edificio histórico",
            " que es un edificio historico",
            ", es un edificio histórico",
            ", es un edificio historico",
        ]:
            if query.lower().endswith(suffix):
                query = query[: -len(suffix)]
                break

        return query.strip(" .,:;")

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

    def _grounded_tourism_candidate_reply(self, message: str, places: list[POI]) -> str:
        first = places[0]
        second = places[1] if len(places) > 1 else None
        lowered = clean_text(message).lower()

        if any(token in lowered for token in ["localizar", "ubicar", "indicarme donde esta", "indicarme dónde está", "mapa", "frente"]):
            return (
                f"Sí, es muy probable que sea {first.name}. "
                "Te lo he marcado en el mapa para que lo ubiques mejor y puedas ubicarlo al llegar."
            )

        if any(token in lowered for token in ["que es", "qué es", "háblame", "hablame"]):
            if second:
                return (
                    f"Sí, te diría que empieces por {first.name}. "
                    f"{self._poi_reason(first)} Si quieres comparar con algo parecido, también tienes {second.name}."
                )
            return f"Sí, te diría que empieces por {first.name}. {self._poi_reason(first)}"

        if second:
            return (
                f"Si te apetece ese plan, probaría primero con {first.name}. "
                f"{self._poi_reason(first)} Como alternativa cercana, también miraría {second.name}."
            )
        return f"Si te apetece ese plan, probaría con {first.name}. {self._poi_reason(first)}"

    def _grounded_ephemeral_map_reply(self, message: str, places: list[POI]) -> str:
        first = places[0]
        lowered = clean_text(message).lower()

        if self._asks_about_map_presence(message) and not self._references_external_map(message):
            return (
                f"Sí. En el mapa de Locus lo vas a encontrar ya marcado como recomendacion temporal, y en este caso el punto es {first.name}. "
                "Cuando llegues a la zona, te servira para ubicarlo sin tener que buscarlo aparte."
            )

        if any(token in lowered for token in ["donde esta", "dónde está", "ubicar", "localizar", "indicarme"]):
            return (
                f"Sí, te lo he dejado marcado en el mapa de Locus como {first.name}. "
                "Lo veras en la zona inmediata a la que te estoy guiando."
            )

        return f"Sí. Lo tienes ya marcado en el mapa de Locus como recomendacion temporal: {first.name}."

    def _poi_reason(self, poi: POI) -> str:
        text = clean_text(poi.description or poi.summary)
        if not text:
            return "Es una opcion clara de las que tienes ya visibles en el mapa."
        if poi.is_ephemeral and any(char.isdigit() for char in text):
            return "Te encaja como parada concreta y razonable para ese plan."
        first_sentence = text.split(".")[0].strip()
        if first_sentence and len(first_sentence) > 8:
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

    def _reply_mentions_specific_poi(self, reply: str, poi: POI) -> bool:
        return self._poi_match_score(reply, poi) > 0

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

    def _reply_sounds_generic_for_focus(self, reply: str) -> bool:
        lowered = clean_text(reply).lower()
        generic_markers = [
            "para orientarte con lo que ya tienes delante",
            "si me dices el plan que te apetece",
            "me fijaria en",
            "me fijaría en",
            "seguimos con",
            "tambien tienes cerca",
            "también tienes cerca",
        ]
        return any(marker in lowered for marker in generic_markers)

    def _focused_poi_reply(self, message: str, poi: POI) -> str:
        lowered = clean_text(message).lower()
        reason = self._poi_reason(poi)

        if any(token in lowered for token in ["antes de comer", "hora de comer", "organizar", "ya", "plan"]):
            return (
                f"Sí: si te interesa {poi.name}, yo iría ya a por esa visita. {reason} "
                "Hazla a tu ritmo y luego rematas la zona antes de comer sin complicarte."
            )

        if any(token in lowered for token in ["quiero visitar", "quiero ver", "me interesa", "tengo mucho interes", "tengo mucho interés"]):
            return (
                f"Sí: si te apetece {poi.name}, la tomaría como tu siguiente parada. {reason} "
                "Si quieres, te la ordeno en versión rápida o tranquila."
            )

        return f"Sí: {poi.name} encaja bien con lo que buscas ahora. {reason}"

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
        named_focus = self._find_named_focus_poi(message, session)

        if named_focus is not None:
            return self._focused_poi_reply(message, named_focus)

        if lowered.strip() in {"hola", "buenas", "hey", "holi", "hello"}:
            return "Hola. Estoy contigo. Dime qué te apetece hacer y te ayudo a orientarte por aquí."

        if visible:
            if any(token in lowered for token in ["cans", "45", "poco", "cerca", "andar", "rapido", "rápido", "caminar"]):
                return self._grounded_map_reply(message, visible)

            if active_name:
                return f"Seguimos con {active_name}. Si quieres, te ayudo a situarte mejor o a organizar esa visita."

            return "Estoy contigo. Si quieres ver algo concreto o montar un plan desde donde estás, te ayudo."

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

    def _run_landmark_identification(self, session, message: str) -> list[POI]:
        reference_text = self._extract_landmark_reference_text(session, message)
        if session.active_poi is None or not reference_text:
            return []

        raw_output = tool_runtime_service.execute(
            session.session_id,
            "identify_map_landmark",
            {
                "reference_text": reference_text,
                "near_poi_name": session.active_poi.name,
                "lat": session.location.lat,
                "lng": session.location.lng,
                "limit": 5,
            },
        )
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return []
        if not payload.get("ok"):
            return []
        identified = [POI(**item) for item in payload.get("pois", [])]
        if not identified:
            return []

        mark_output = tool_runtime_service.execute(
            session.session_id,
            "mark_pois_on_map",
            {
                "poi_names": [identified[0].name],
                "reason": "Landmark identified near active POI and should be visible on the Locus map",
            },
        )
        try:
            mark_payload = json.loads(mark_output)
        except json.JSONDecodeError:
            return identified
        if not mark_payload.get("ok"):
            return identified
        return [POI(**item) for item in mark_payload.get("marked_pois", [])]

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
        try:
            if self.openai.is_configured():
                if session.user_id is not None:
                    billing_service.ensure_user_can_consume(session.user_id)
                reply, final_response = self._run_openai_chat(session.session_id, clean_message)
                final_session = session_service.get_or_create(session.session_id)
                ephemeral_pois = list(final_session.ephemeral_map_pois)
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
