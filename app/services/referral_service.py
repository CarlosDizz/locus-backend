from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

from app.clients.openai_client import OpenAIClient, OpenAIClientError
from app.config import settings
from app.schemas.catalog import PoiResponse
from app.services.session_service import session_service
from app.utils.logging import get_logger
from app.utils.text import clean_text


@dataclass(frozen=True)
class AccessReferralLink:
    title: str
    description: str
    url: str
    kind: str
    query: str
    provider: str = "getyourguide"
    tracking_status: str = "untracked"


class ReferralService:
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    ticket_terms = [
        "museo",
        "museum",
        "catedral",
        "cathedral",
        "palacio",
        "palace",
        "alcázar",
        "alcazar",
        "castillo",
        "castle",
        "monasterio",
        "monastery",
        "basílica",
        "basilica",
        "yacimiento",
        "arqueológico",
        "archaeological",
        "anfiteatro",
        "teatro romano",
        "mirador",
        "observatory",
        "tower",
        "torre",
    ]
    attraction_terms = [
        "parque",
        "aquarium",
        "acuario",
        "zoo",
        "teleférico",
        "teleferico",
        "mirador",
        "atracción",
        "attraction",
    ]
    mobility_terms = [
        "bus turístico",
        "bus turistico",
        "barco",
        "boat",
        "crucero",
        "cruise",
        "tren turístico",
        "tren turistico",
        "segway",
        "quad",
        "buggy",
        "helicóptero",
        "helicoptero",
    ]
    guided_terms = [
        "free tour",
        "tour a pie",
        "walking tour",
        "visita guiada",
        "guía privado",
        "guia privado",
        "guided tour",
    ]

    def poi_access_links(self, poi: PoiResponse, *, city_name: str = "") -> dict:
        profile = self._infer_poi_access_profile(poi)
        links = self._curated_links_from_metadata(poi.metadata or {}, fallback_title=poi.name)
        if not links:
            return {
                "poi_id": poi.id,
                "poi_name": poi.name,
                "eligible": False,
                "reason": (
                    "No hay enlace de entrada o acceso curado para este POI. "
                    "Mejor consultar fuente oficial antes que mostrar una busqueda generica."
                ),
                "links": [],
            }

        return {
            "poi_id": poi.id,
            "poi_name": poi.name,
            "eligible": True,
            "reason": profile["reason"] if profile else "Este POI tiene enlaces de acceso curados.",
            "links": [link.__dict__ for link in links],
        }

    def activity_referrals(
        self,
        *,
        session_id: str,
        query: str,
        poi_name: str = "",
        city_name: str = "",
        intent: str = "",
        max_results: int = 3,
    ) -> dict:
        self._log(
            "tool_start",
            session_id=session_id,
            query=query,
            poi_name=poi_name,
            city_name=city_name,
            intent=intent,
            max_results=max_results,
        )
        if not settings.getyourguide_referrals_enabled:
            self._log("tool_disabled", session_id=session_id)
            return {
                "ok": False,
                "error": "referrals_disabled",
                "message": "La busqueda de entradas y experiencias esta desactivada.",
            }

        session = session_service.get_or_create(session_id)
        clean_query = clean_text(query)
        clean_poi = clean_text(poi_name or (session.active_poi.name if session.active_poi else ""))
        clean_city = clean_text(city_name)
        search_text = self._compose_search_text(clean_query, clean_poi, clean_city)
        self._log("tool_context", session_id=session_id, search_text=search_text, clean_poi=clean_poi, clean_city=clean_city)
        if not search_text:
            self._log("tool_error", session_id=session_id, error="query_required")
            return {"ok": False, "error": "query_required", "message": "Falta una busqueda concreta."}

        if session.active_poi is not None:
            curated_links = self._curated_links_from_metadata(
                getattr(session.active_poi, "metadata", None) or {},
                fallback_title=session.active_poi.name,
            )
            if curated_links:
                self._log("tool_curated_links", session_id=session_id, count=len(curated_links))
                return {
                    "ok": True,
                    "provider": "curated",
                    "query": search_text,
                    "links": [link.__dict__ for link in curated_links[: max(1, min(max_results, 5))]],
                    "policy": "Estos son enlaces curados. Presentalos como acceso concreto, no como busqueda generica.",
                }

        if self._looks_like_guided_visit(search_text) and not self._looks_like_non_substitutable_experience(search_text):
            self._log("tool_rejected", session_id=session_id, error="guided_visit_competes_with_locus", search_text=search_text)
            return {
                "ok": False,
                "error": "guided_visit_competes_with_locus",
                "message": "No recomiendes visitas guiadas culturales normales: esa experiencia la cubre Locus. Ofrece entradas, pases o transporte si aplica.",
            }

        web_links = self._search_getyourguide_product_links(
            query=search_text,
            poi_name=clean_poi,
            city_name=clean_city,
            intent=intent,
            max_results=max_results,
        )
        if web_links:
            self._log("tool_success", session_id=session_id, provider="getyourguide_websearch", count=len(web_links))
            return {
                "ok": True,
                "provider": "getyourguide_websearch",
                "query": search_text,
                "links": [link.__dict__ for link in web_links],
                "policy": (
                    "Estos enlaces vienen de paginas concretas encontradas en GetYourGuide mediante busqueda web. "
                    "Presentalos por titulo, no como busqueda. Si dudas de encaje, ofrece tambien contrastar la web oficial."
                ),
            }

        self._log("tool_no_links", session_id=session_id, query=search_text)
        return {
            "ok": False,
            "error": "no_reliable_referral_link",
            "provider": "curated",
            "query": search_text,
            "links": [],
            "message": (
                "No tengo un enlace concreto y fiable de entrada/pase para este lugar. "
                "No muestres busquedas genericas de GetYourGuide como si fueran entradas. "
                "Consulta informacion oficial o busqueda web para precio, horarios y venta online."
            ),
        }

    def _infer_poi_access_profile(self, poi: PoiResponse) -> dict | None:
        haystack = clean_text(
            " ".join(
                [
                    poi.name,
                    poi.poi_type_code or "",
                    poi.poi_type_name or "",
                    poi.short_description,
                    poi.long_description,
                ]
            )
        ).lower()
        metadata = poi.metadata or {}
        if metadata.get("access_referrals") is False:
            return None
        if metadata.get("access_referrals") is True:
            return {
                "kind": "ticket",
                "query_prefix": "entradas",
                "reason": "Este POI esta marcado como reservable en el catalogo.",
            }
        if any(term in haystack for term in self.ticket_terms):
            return {
                "kind": "ticket",
                "query_prefix": "entradas",
                "reason": "Este tipo de lugar puede requerir entrada, reserva o acceso controlado.",
            }
        if any(term in haystack for term in self.attraction_terms):
            return {
                "kind": "pass",
                "query_prefix": "pases entradas",
                "reason": "Esta atraccion puede requerir pase, entrada o reserva.",
            }
        if any(term in haystack for term in self.mobility_terms):
            return {
                "kind": "transport",
                "query_prefix": "tickets",
                "reason": "Esta experiencia depende de transporte o acceso fisico reservado.",
            }
        return None

    def _curated_links_from_metadata(self, metadata: dict, *, fallback_title: str) -> list[AccessReferralLink]:
        raw_links = metadata.get("access_links") or metadata.get("ticket_links") or []
        if not isinstance(raw_links, list):
            raw_links = []
        single_url = clean_text(str(metadata.get("ticket_url") or metadata.get("official_ticket_url") or metadata.get("getyourguide_url") or ""))
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
        if provider != "getyourguide" or not settings.getyourguide_partner_id:
            return url
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("partner_id", settings.getyourguide_partner_id)
        query.setdefault("utm_medium", "travel_agent")
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _search_getyourguide_product_links(
        self,
        *,
        query: str,
        poi_name: str,
        city_name: str,
        intent: str,
        max_results: int,
    ) -> list[AccessReferralLink]:
        client = OpenAIClient()
        if not client.is_configured():
            self._log("websearch_skipped", reason="openai_not_configured", query=query)
            return []

        web_tool: dict = {
            "type": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "ES",
                "timezone": "Europe/Madrid",
            },
            "filters": {"allowed_domains": ["getyourguide.es", "getyourguide.com"]},
        }
        try:
            self._log("websearch_start", query=query, poi_name=poi_name, city_name=city_name, intent=intent)
            response = client.create_response(
                model=client.chat_model(),
                instructions=(
                    "Busca solo paginas concretas de producto en GetYourGuide para entradas, pases, tickets, "
                    "transporte turistico o experiencias fisicas. No devuelvas paginas de busqueda, categorias, "
                    "ciudades ni articulos. La ciudad indicada es obligatoria: si el producto no esta en esa ciudad "
                    "o no menciona claramente el lugar buscado, no sirve. Evita visitas guiadas culturales normales "
                    "si no aportan acceso fisico."
                ),
                input_items=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Encuentra productos concretos de GetYourGuide para: {query}\n"
                                    f"Ciudad obligatoria: {city_name or '(sin ciudad)'}\n"
                                    f"Lugar/POI obligatorio: {poi_name or query}\n"
                                    f"Intencion: {intent or 'access'}\n"
                                    "Prioriza paginas de producto con entrada, pase, ticket, acceso sin cola, bus, barco, "
                                    "tren turistico, teleferico o transporte. No devuelvas experiencias generales de la ciudad "
                                    "si no son para el lugar/POI pedido. Si no hay producto claro de ese lugar en esa ciudad, no inventes."
                                ),
                            }
                        ],
                    }
                ],
                tools=[web_tool],
                max_output_tokens=220,
                extra_payload={"include": ["web_search_call.action.sources"]},
            )
        except OpenAIClientError as exc:
            self._log("websearch_failed", query=query, error=str(exc))
            return []

        candidates = self._extract_web_sources(response)
        self._log("websearch_sources", query=query, count=len(candidates), sources=[source.get("url", "") for source in candidates[:8]])
        links: list[AccessReferralLink] = []
        seen_urls: set[str] = set()
        for source in candidates:
            url = clean_text(source.get("url", ""))
            title = clean_text(source.get("title", ""))
            if not self._is_getyourguide_product_url(url):
                self._log("candidate_rejected", reason="not_product_url", title=title, url=url)
                continue
            if not self._source_matches_place(title=title, url=url, poi_name=poi_name, city_name=city_name, query=query):
                self._log("candidate_rejected", reason="place_mismatch", title=title, url=url, poi_name=poi_name, city_name=city_name)
                continue
            if self._looks_like_guided_visit(f"{title} {unquote(url)}"):
                self._log("candidate_rejected", reason="guided_visit", title=title, url=url)
                continue
            final_url = self._decorate_url(url, provider="getyourguide")
            if final_url in seen_urls:
                self._log("candidate_rejected", reason="duplicate", title=title, url=url)
                continue
            seen_urls.add(final_url)
            self._log("candidate_accepted", title=title or self._title_from_url(url), url=final_url)
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
        return links

    def _log(self, event: str, **payload) -> None:
        message = "referral_tool " + " ".join(f"{key}={value!r}" for key, value in payload.items())
        self.logger.warning("%s %s", event, message)
        print(f"{event} {message}", flush=True)

    def _extract_web_sources(self, response: dict) -> list[dict[str, str]]:
        collected: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in response.get("output", []) or []:
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                for source in action.get("sources", []) or []:
                    url = clean_text(str(source.get("url") or ""))
                    if url and url not in seen:
                        seen.add(url)
                        collected.append({"title": clean_text(str(source.get("title") or "")), "url": url})
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                for annotation in content.get("annotations", []) or []:
                    if annotation.get("type") != "url_citation":
                        continue
                    url = clean_text(str(annotation.get("url") or ""))
                    if url and url not in seen:
                        seen.add(url)
                        collected.append({"title": clean_text(str(annotation.get("title") or "")), "url": url})
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
        if place_tokens and not self._has_enough_place_overlap(text, place_tokens):
            return False
        return True

    def _city_aliases(self, city_name: str) -> list[str]:
        city = self._normalize_search_text(city_name)
        if not city:
            return []
        aliases = {
            "roma": ["roma", "rome"],
            "florencia": ["florencia", "florence", "firenze"],
            "venecia": ["venecia", "venice", "venezia"],
            "napoles": ["napoles", "napoli", "naples"],
            "milan": ["milan", "milano"],
        }
        return aliases.get(city, [city])

    def _meaningful_tokens(self, text: str) -> list[str]:
        stopwords = {
            "entrada", "entradas", "ticket", "tickets", "pase", "pases", "precio", "precios",
            "tarifa", "tarifas", "comprar", "reserva", "reservar", "museo", "municipal",
            "de", "del", "la", "el", "los", "las", "para", "por", "con", "sin", "y", "en",
        }
        normalized = self._normalize_search_text(text)
        return [
            token
            for token in re.split(r"[^a-z0-9]+", normalized)
            if len(token) >= 4 and token not in stopwords
        ][:8]

    def _has_enough_place_overlap(self, text: str, place_tokens: list[str]) -> bool:
        if not place_tokens:
            return True
        matches = [token for token in place_tokens if token in text]
        if len(place_tokens) == 1:
            return bool(matches)
        required = 2 if len(place_tokens) >= 2 else 1
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
        if any(term in lowered for term in self.mobility_terms):
            return "transport"
        if any(term in lowered for term in self.attraction_terms):
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

    def _looks_like_guided_visit(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in self.guided_terms)

    def _looks_like_non_substitutable_experience(self, text: str) -> bool:
        lowered = text.lower()
        allowed_terms = [
            "entrada",
            "ticket",
            "pase",
            "reserva",
            "skip the line",
            "sin cola",
            *self.attraction_terms,
            *self.mobility_terms,
        ]
        return any(term in lowered for term in allowed_terms)


referral_service = ReferralService()
