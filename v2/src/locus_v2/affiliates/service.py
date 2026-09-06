"""GetYourGuide affiliate referrals, ported from app/services/referral_service.py.

Real live revenue in V1 (`app/services/referral_service.py`, used by
`/catalog/pois/{id}/access-links` and the `activity_referrals`/`find_activities`
tool). Async (httpx-free — uses the OpenAI SDK's own async client directly,
matching voice/tools.py) instead of V1's sync OpenAI wrapper.

Depends only on the Sessions domain (for `active_poi`/session lookup) and
Settings — both already in place, so this stood on its own without needing
the Chat domain or a public Catalog API first. What DOES still depend on the
(not yet built) public Catalog API is exposing `poi_access_links` over HTTP
at `/catalog/pois/{id}/access-links` — that route isn't written yet. This
service is ready for it, and is already wired into the voice tool dispatcher
(`voice/tools.py`) as `affiliates.find_activities`, usable today from any
real call via the POI test harness in the control panel.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import structlog
from openai import AsyncOpenAI

from locus_v2.config import Settings
from locus_v2.sessions.application.service import MapSessionService
from locus_v2.shared.openai_usage import ToolUsage, usage_from_openai_response
from locus_v2.shared.text import clean_text

logger = structlog.get_logger()

TICKET_TERMS = [
    "museo", "museum", "catedral", "cathedral", "palacio", "palace", "alcázar", "alcazar",
    "castillo", "castle", "monasterio", "monastery", "basílica", "basilica", "yacimiento",
    "arqueológico", "archaeological", "anfiteatro", "teatro romano", "mirador",
    "observatory", "tower", "torre",
]
ATTRACTION_TERMS = [
    "parque", "aquarium", "acuario", "zoo", "teleférico", "teleferico", "mirador",
    "atracción", "attraction",
]
MOBILITY_TERMS = [
    "bus turístico", "bus turistico", "barco", "boat", "crucero", "cruise",
    "tren turístico", "tren turistico", "segway", "quad", "buggy", "helicóptero",
    "helicoptero",
]
GUIDED_TERMS = [
    "free tour", "tour a pie", "walking tour", "visita guiada", "visita en grupo",
    "guía privado", "guia privado", "guia privada", "guia exclusiva", "guided tour",
    "tour sin colas", "tour expres",
]
_CITY_ALIASES = {
    "roma": ["roma", "rome"],
    "florencia": ["florencia", "florence", "firenze"],
    "venecia": ["venecia", "venice", "venezia"],
    "napoles": ["napoles", "napoli", "naples"],
    "milan": ["milan", "milano"],
}
_PLACE_TOKEN_ALIASES = {
    "coliseo": ["coliseo", "colosseo", "colosseum", "coliseum"],
    "colosseo": ["coliseo", "colosseo", "colosseum", "coliseum"],
    "colosseum": ["coliseo", "colosseo", "colosseum", "coliseum"],
    "coliseum": ["coliseo", "colosseo", "colosseum", "coliseum"],
    "foro": ["foro", "forum"],
    "forum": ["foro", "forum"],
    "palatino": ["palatino", "palatine"],
    "palatine": ["palatino", "palatine"],
    "maggiore": ["maggiore", "mayor"],
    "quirinale": ["quirinale", "quirinal"],
}
_STOPWORDS = {
    "entrada", "entradas", "ticket", "tickets", "pase", "pases", "precio", "precios",
    "tarifa", "tarifas", "comprar", "reserva", "reservar", "museo", "municipal",
    "de", "del", "la", "el", "los", "las", "para", "por", "con", "sin", "y", "en",
}
_IGNORED_TRACKING_KEYS = {
    "visitor-id", "utm_source", "utm_campaign", "utm_content", "utm_term", "cmp",
    "currency", "psrc", "partner_id",
}


@dataclass(frozen=True)
class AccessReferralLink:
    title: str
    description: str
    url: str
    kind: str
    query: str
    provider: str = "getyourguide"
    tracking_status: str = "untracked"

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "kind": self.kind,
            "query": self.query,
            "provider": self.provider,
            "tracking_status": self.tracking_status,
        }


class ReferralService:
    def __init__(self, settings: Settings, session_service: MapSessionService | None = None) -> None:
        self.settings = settings
        self.session_service = session_service

    def poi_access_links(
        self,
        *,
        poi_id: str,
        poi_name: str,
        poi_type_code: str,
        poi_type_name: str,
        short_description: str,
        long_description: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        profile = self._infer_poi_access_profile(
            poi_name=poi_name,
            poi_type_code=poi_type_code,
            poi_type_name=poi_type_name,
            short_description=short_description,
            long_description=long_description,
            metadata=metadata,
        )
        links = self._curated_links_from_metadata(metadata, fallback_title=poi_name)
        if not links:
            return {
                "poi_id": poi_id,
                "poi_name": poi_name,
                "eligible": False,
                "reason": (
                    "No hay enlace de entrada o acceso curado para este POI. "
                    "Mejor consultar fuente oficial antes que mostrar una busqueda generica."
                ),
                "links": [],
            }
        return {
            "poi_id": poi_id,
            "poi_name": poi_name,
            "eligible": True,
            "reason": profile["reason"] if profile else "Este POI tiene enlaces de acceso curados.",
            "links": [link.to_dict() for link in links],
        }

    async def activity_referrals(
        self,
        *,
        session_id: str,
        query: str,
        poi_name: str = "",
        city_name: str = "",
        intent: str = "",
        max_results: int = 3,
    ) -> dict[str, Any]:
        if not self.settings.getyourguide_referrals_enabled:
            return {
                "ok": False,
                "error": "referrals_disabled",
                "message": "La busqueda de entradas y experiencias esta desactivada.",
            }

        active_poi_name = ""
        active_poi_metadata: dict[str, Any] = {}
        if self.session_service is not None:
            session = await self.session_service.get_or_create(session_id)
            if session.active_poi is not None:
                active_poi_name = session.active_poi.name
                active_poi_metadata = {}  # SessionPoi has no metadata field, see docstring note

        clean_query = clean_text(query)
        clean_poi = clean_text(poi_name or active_poi_name)
        clean_city = clean_text(city_name)
        search_text = self._compose_search_text(clean_query, clean_poi, clean_city)
        if not search_text:
            return {"ok": False, "error": "query_required", "message": "Falta una busqueda concreta."}

        if active_poi_metadata:
            curated_links = self._curated_links_from_metadata(active_poi_metadata, fallback_title=clean_poi)
            if curated_links:
                return {
                    "ok": True,
                    "provider": "curated",
                    "query": search_text,
                    "links": [link.to_dict() for link in curated_links[: max(1, min(max_results, 5))]],
                    "policy": (
                        "Estos son enlaces curados. Presentalos como acceso concreto, no como busqueda generica. "
                        "Usa enlaces Markdown clicables con titulo humano: [titulo](url)."
                    ),
                }

        web_links, usage = await self._search_getyourguide_product_links(
            query=search_text, poi_name=clean_poi, city_name=clean_city, intent=intent, max_results=max_results
        )
        # The web_search call above costs real tokens whether or not it finds a link (the
        # fallback branch below still made the same call) - _usage rides along on every
        # return path so the caller can bill it. VoiceToolDispatcher pops this key before
        # the result ever reaches the model.
        if web_links:
            return {
                "ok": True,
                "provider": "getyourguide_websearch",
                "query": search_text,
                "links": [link.to_dict() for link in web_links],
                "policy": (
                    "Estos enlaces vienen de paginas concretas encontradas en GetYourGuide mediante busqueda web. "
                    "Presentalos como enlaces Markdown clicables con titulo humano: [titulo](url), no como busqueda. "
                    "No uses backticks para sustituir el enlace. Si dudas de encaje, ofrece tambien contrastar la web oficial."
                ),
                "_usage": usage,
            }

        fallback_link = self._fallback_getyourguide_search_link(search_text, poi_name=clean_poi, city_name=clean_city)
        return {
            "ok": True,
            "provider": "getyourguide_fallback_search",
            "query": search_text,
            "links": [fallback_link.to_dict()],
            "policy": (
                "No se encontro un producto concreto suficientemente fiable. Este enlace es una busqueda sugerida "
                "de GetYourGuide, no una entrada oficial ni una recomendacion garantizada. Presentalo como opcion "
                "secundaria con lenguaje honesto: 'ver opciones en GetYourGuide'."
            ),
            "_usage": usage,
        }

    def _infer_poi_access_profile(
        self,
        *,
        poi_name: str,
        poi_type_code: str,
        poi_type_name: str,
        short_description: str,
        long_description: str,
        metadata: dict[str, Any],
    ) -> dict[str, str] | None:
        haystack = clean_text(
            " ".join([poi_name, poi_type_code or "", poi_type_name or "", short_description, long_description])
        ).lower()
        if metadata.get("access_referrals") is False:
            return None
        if metadata.get("access_referrals") is True:
            return {
                "kind": "ticket", "query_prefix": "entradas",
                "reason": "Este POI esta marcado como reservable en el catalogo.",
            }
        if any(term in haystack for term in TICKET_TERMS):
            return {
                "kind": "ticket", "query_prefix": "entradas",
                "reason": "Este tipo de lugar puede requerir entrada, reserva o acceso controlado.",
            }
        if any(term in haystack for term in ATTRACTION_TERMS):
            return {
                "kind": "pass", "query_prefix": "pases entradas",
                "reason": "Esta atraccion puede requerir pase, entrada o reserva.",
            }
        if any(term in haystack for term in MOBILITY_TERMS):
            return {
                "kind": "transport", "query_prefix": "tickets",
                "reason": "Esta experiencia depende de transporte o acceso fisico reservado.",
            }
        return None

    def _curated_links_from_metadata(
        self, metadata: dict[str, Any], *, fallback_title: str
    ) -> list[AccessReferralLink]:
        raw_links = metadata.get("access_links") or metadata.get("ticket_links") or []
        if not isinstance(raw_links, list):
            raw_links = []
        single_url = clean_text(
            str(metadata.get("ticket_url") or metadata.get("official_ticket_url") or metadata.get("getyourguide_url") or "")
        )
        if single_url:
            raw_links = [
                *raw_links,
                {
                    "title": f"Entradas para {fallback_title}",
                    "description": "Enlace de acceso curado para este lugar.",
                    "url": single_url,
                    "kind": "ticket",
                    "provider": "getyourguide" if "getyourguide." in single_url else "official",
                },
            ]

        links: list[AccessReferralLink] = []
        seen_urls: set[str] = set()
        for item in raw_links:
            if not isinstance(item, dict):
                continue
            url = clean_text(str(item.get("url") or ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            provider = clean_text(str(item.get("provider") or ("getyourguide" if "getyourguide." in url else "official")))
            final_url = self._decorate_url(url, provider=provider)
            links.append(
                AccessReferralLink(
                    title=clean_text(str(item.get("title") or f"Entradas para {fallback_title}")),
                    description=clean_text(str(item.get("description") or "Consulta acceso, precio y disponibilidad.")),
                    url=final_url,
                    kind=clean_text(str(item.get("kind") or "ticket")),
                    query="",
                    provider=provider,
                    tracking_status="tracked" if provider == "getyourguide" and "partner_id=" in final_url else "official",
                )
            )
        return links

    def _decorate_url(self, url: str, *, provider: str) -> str:
        if provider != "getyourguide":
            return url
        parsed = urlparse(url)
        query = {
            key: value
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _IGNORED_TRACKING_KEYS
        }
        if self.settings.getyourguide_partner_id:
            query["partner_id"] = self.settings.getyourguide_partner_id
            query["utm_medium"] = "travel_agent"
        return urlunparse(parsed._replace(query=urlencode(query)))

    async def _search_getyourguide_product_links(
        self, *, query: str, poi_name: str, city_name: str, intent: str, max_results: int
    ) -> tuple[list[AccessReferralLink], ToolUsage]:
        if self.settings.openai_api_key is None:
            return [], ToolUsage()
        client = AsyncOpenAI(api_key=self.settings.openai_api_key.get_secret_value())
        try:
            response = await client.responses.create(
                model=self.settings.tool_model,
                instructions=(
                    "Busca paginas concretas de producto en GetYourGuide para entradas, pases, tickets, tours, "
                    "free tours, visitas guiadas, excursiones, transporte turistico o experiencias reservables. "
                    "No devuelvas paginas de busqueda, categorias, ciudades ni articulos. La ciudad indicada es "
                    "importante: si el producto claramente no esta en esa ciudad o no encaja con el lugar buscado, no sirve."
                ),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Encuentra productos concretos de GetYourGuide para: {query}\n"
                                    f"Ciudad obligatoria: {city_name or '(sin ciudad)'}\n"
                                    f"Lugar/POI obligatorio: {poi_name or query}\n"
                                    f"Intencion: {intent or 'activity'}\n"
                                    "Prioriza paginas de producto con entrada, pase, ticket, acceso sin cola, bus, "
                                    "barco, tren turistico, teleferico, transporte, tour, free tour, visita guiada, "
                                    "excursion o experiencia. No inventes productos si no aparecen fuentes claras."
                                ),
                            }
                        ],
                    }
                ],
                tools=[
                    {
                        "type": "web_search",
                        "user_location": {"type": "approximate", "country": "ES", "timezone": "Europe/Madrid"},
                        "filters": {"allowed_domains": ["getyourguide.es", "getyourguide.com"]},
                    }
                ],
                # V1 used 220 here, tuned for a non-reasoning model. locus_v2's tool_model
                # (gpt-5-mini) is a reasoning model that spends hidden reasoning tokens
                # before ever issuing the web_search call, so 220 always got cut off with
                # zero results (confirmed: status="incomplete", no web_search_call emitted
                # at all). 1500 leaves enough room for reasoning + the actual search call.
                max_output_tokens=1500,
                # Same fix as voice/tools.py._ask_model(): unset, this model defaults to a
                # much higher reasoning effort than a "find matching product pages" lookup
                # needs, which was eating into the 1500-token budget the comment above was
                # already tuned for.
                reasoning={"effort": "low"},
                include=["web_search_call.action.sources"],
            )
        except Exception as error:
            logger.warning("referral_websearch_failed", query=query, error=str(error))
            return [], ToolUsage()
        finally:
            await client.close()

        usage = usage_from_openai_response(response)
        candidates = self._extract_web_sources(response)
        links: list[AccessReferralLink] = []
        seen_urls: set[str] = set()
        for source in candidates:
            url = clean_text(source.get("url", ""))
            title = clean_text(source.get("title", ""))
            if not self._is_getyourguide_product_url(url):
                continue
            if not self._source_matches_place(title=title, url=url, poi_name=poi_name, city_name=city_name, query=query):
                continue
            final_url = self._decorate_url(url, provider="getyourguide")
            if final_url in seen_urls:
                continue
            seen_urls.add(final_url)
            links.append(
                AccessReferralLink(
                    title=title or self._title_from_url(url),
                    description="Producto concreto encontrado en GetYourGuide.",
                    url=final_url,
                    kind=self._infer_link_kind(f"{intent} {title} {url}"),
                    query=query,
                    provider="getyourguide",
                    tracking_status="tracked" if "partner_id=" in final_url else "untracked",
                )
            )
            if len(links) >= max(1, min(max_results, 5)):
                break
        return links, usage

    def _fallback_getyourguide_search_link(
        self, query: str, *, poi_name: str, city_name: str
    ) -> AccessReferralLink:
        search_query = self._compose_search_text(query, poi_name, city_name)
        url = self._decorate_url(
            "https://www.getyourguide.es/s/?" + urlencode({"q": search_query}), provider="getyourguide"
        )
        title_subject = poi_name or city_name or query
        return AccessReferralLink(
            title=f"Ver opciones para {title_subject} en GetYourGuide",
            description="Busqueda sugerida de actividades, tours o entradas. Revisa encaje, precio y disponibilidad.",
            url=url,
            kind="activity",
            query=search_query,
            provider="getyourguide",
            tracking_status="fallback_search_tracked" if "partner_id=" in url else "fallback_search",
        )

    def _extract_web_sources(self, response: Any) -> list[dict[str, str]]:
        collected: list[dict[str, str]] = []
        seen: set[str] = set()
        output = getattr(response, "output", None) or []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                action = getattr(item, "action", None)
                sources = getattr(action, "sources", None) or []
                for source in sources:
                    url = clean_text(str(getattr(source, "url", "") or ""))
                    if url and url not in seen:
                        seen.add(url)
                        collected.append({"title": clean_text(str(getattr(source, "title", "") or "")), "url": url})
            if item_type != "message":
                continue
            for content in getattr(item, "content", None) or []:
                for annotation in getattr(content, "annotations", None) or []:
                    if getattr(annotation, "type", None) != "url_citation":
                        continue
                    url = clean_text(str(getattr(annotation, "url", "") or ""))
                    if url and url not in seen:
                        seen.add(url)
                        collected.append({"title": clean_text(str(getattr(annotation, "title", "") or "")), "url": url})
        return collected

    def _is_getyourguide_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        if "getyourguide." not in host:
            return False
        if path.startswith("/s/") or path in {"", "/", "/es-es/"}:
            return False
        if "/c/" in path or re.search(r"/[a-z]{2}(?:-[a-z]{2})?/(?:all-activities|things-to-do|s)/", path):
            return False
        return bool(re.search(r"-t\d+(?:/|$)", path))

    def _source_matches_place(self, *, title: str, url: str, poi_name: str, city_name: str, query: str) -> bool:
        text = self._normalize_search_text(f"{title} {unquote(url)}")
        city_aliases = self._city_aliases(city_name)
        if city_aliases and not any(alias in text for alias in city_aliases):
            return False
        place_tokens = self._meaningful_tokens(poi_name) or self._meaningful_tokens(query)
        expanded_place_tokens = self._expand_place_tokens(place_tokens)
        if place_tokens and not self._has_enough_place_overlap(
            text, expanded_place_tokens, original_token_count=len(place_tokens)
        ):
            return False
        return True

    def _city_aliases(self, city_name: str) -> list[str]:
        city = self._normalize_search_text(city_name)
        if not city:
            return []
        return _CITY_ALIASES.get(city, [city])

    def _meaningful_tokens(self, text: str) -> list[str]:
        normalized = self._normalize_search_text(text)
        return [
            token for token in re.split(r"[^a-z0-9]+", normalized)
            if len(token) >= 4 and token not in _STOPWORDS
        ][:8]

    def _expand_place_tokens(self, tokens: list[str]) -> list[str]:
        expanded: list[str] = []
        for token in tokens:
            for alias in _PLACE_TOKEN_ALIASES.get(token, [token]):
                if alias not in expanded:
                    expanded.append(alias)
        return expanded

    def _has_enough_place_overlap(
        self, text: str, place_tokens: list[str], *, original_token_count: int | None = None
    ) -> bool:
        if not place_tokens:
            return True
        matches = [token for token in place_tokens if token in text]
        token_count = original_token_count or len(place_tokens)
        if token_count == 1:
            return bool(matches)
        required = 2 if token_count >= 2 else 1
        return len(matches) >= required

    def _normalize_search_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", clean_text(text).lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))

    def _title_from_url(self, url: str) -> str:
        path = unquote(urlparse(url).path)
        slug = path.strip("/").split("/")[-1]
        slug = re.sub(r"-t\d+$", "", slug)
        words = [word for word in slug.split("-") if word]
        if not words:
            return "Entrada o acceso en GetYourGuide"
        return " ".join(words[:12]).capitalize()

    def _infer_link_kind(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in MOBILITY_TERMS):
            return "transport"
        if any(term in lowered for term in ATTRACTION_TERMS):
            return "pass"
        return "ticket"

    def _compose_search_text(self, query: str, poi_name: str, city_name: str) -> str:
        parts: list[str] = []
        for part in [query, poi_name, city_name]:
            clean_part = clean_text(part)
            if not clean_part:
                continue
            lowered_part = clean_part.lower()
            if any(lowered_part in existing.lower() or existing.lower() in lowered_part for existing in parts):
                continue
            parts.append(clean_part)
        return clean_text(" ".join(parts))
