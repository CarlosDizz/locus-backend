from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus, urlencode

from app.config import settings
from app.schemas.catalog import PoiResponse
from app.services.session_service import session_service
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
        if profile is None:
            return {
                "poi_id": poi.id,
                "poi_name": poi.name,
                "eligible": False,
                "reason": "Este POI parece mejor cubierto por la guia de Locus sin compra de acceso.",
                "links": [],
            }

        links = self._build_links(
            base_query=f"{profile['query_prefix']} {poi.name} {city_name}".strip(),
            city_name=city_name,
            limit=2,
            kinds=[profile["kind"], "access"],
        )
        return {
            "poi_id": poi.id,
            "poi_name": poi.name,
            "eligible": True,
            "reason": profile["reason"],
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
        if not settings.getyourguide_referrals_enabled:
            return {
                "ok": False,
                "error": "referrals_disabled",
                "message": "La busqueda de entradas y experiencias esta desactivada.",
            }

        session = session_service.get_or_create(session_id)
        clean_query = clean_text(query)
        clean_poi = clean_text(poi_name or (session.active_poi.name if session.active_poi else ""))
        clean_city = clean_text(city_name)
        search_text = clean_text(" ".join(part for part in [clean_query, clean_poi, clean_city] if part))
        if not search_text:
            return {"ok": False, "error": "query_required", "message": "Falta una busqueda concreta."}

        if self._looks_like_guided_visit(search_text) and not self._looks_like_non_substitutable_experience(search_text):
            return {
                "ok": False,
                "error": "guided_visit_competes_with_locus",
                "message": "No recomiendes visitas guiadas culturales normales: esa experiencia la cubre Locus. Ofrece entradas, pases o transporte si aplica.",
            }

        kinds = self._kinds_for_intent(f"{intent} {search_text}")
        links = self._build_links(
            base_query=search_text,
            city_name=clean_city,
            limit=max_results,
            kinds=kinds,
        )
        return {
            "ok": True,
            "provider": "getyourguide",
            "query": search_text,
            "links": [link.__dict__ for link in links],
            "policy": (
                "Presenta estos enlaces como acceso, pase o experiencia fisica que Locus no puede sustituir. "
                "No los vendas como visita guiada cultural ni los fuerces si el usuario solo quiere contexto."
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

    def _build_links(self, *, base_query: str, city_name: str, limit: int, kinds: list[str]) -> list[AccessReferralLink]:
        templates = {
            "ticket": ("Entradas", "Busca entradas, acceso o reserva para este lugar.", "entradas {query}"),
            "pass": ("Pases", "Opciones de pase o acceso para la atraccion.", "pases entradas {query}"),
            "transport": ("Transporte y experiencias", "Actividades con transporte o experiencia fisica.", "{query} transporte tickets"),
            "access": ("Acceso", "Comprueba disponibilidad antes de ir.", "{query} tickets"),
        }
        normalized_kinds = [kind for kind in kinds if kind in templates]
        if not normalized_kinds:
            normalized_kinds = ["ticket"]

        links: list[AccessReferralLink] = []
        seen_queries: set[str] = set()
        for kind in normalized_kinds:
            title, description, query_template = templates[kind]
            query = clean_text(query_template.format(query=base_query, city=city_name))
            if query.lower() in seen_queries:
                continue
            seen_queries.add(query.lower())
            links.append(
                AccessReferralLink(
                    title=title,
                    description=description,
                    url=self._referral_url(query),
                    kind=kind,
                    query=query,
                    tracking_status="tracked" if self._has_tracking_template() else "untracked",
                )
            )
            if len(links) >= max(1, min(limit, 5)):
                break
        return links

    def _referral_url(self, query: str) -> str:
        if self._has_tracking_template():
            return settings.getyourguide_referral_url_template.format(
                query=quote_plus(query),
                partner_id=quote_plus(settings.getyourguide_partner_id),
            )

        separator = "&" if "?" in settings.getyourguide_search_base_url else "?"
        return f"{settings.getyourguide_search_base_url}{separator}{urlencode({'q': query})}"

    @staticmethod
    def _has_tracking_template() -> bool:
        return bool(settings.getyourguide_referral_url_template and settings.getyourguide_partner_id)

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

    def _kinds_for_intent(self, text: str) -> list[str]:
        lowered = text.lower()
        kinds: list[str] = []
        if any(term in lowered for term in self.mobility_terms):
            kinds.append("transport")
        if any(term in lowered for term in self.attraction_terms):
            kinds.append("pass")
        if any(term in lowered for term in ["entrada", "ticket", "reserva", "museo", "catedral", "palacio", "yacimiento"]):
            kinds.append("ticket")
        if not kinds:
            kinds.append("access")
        return kinds


referral_service = ReferralService()
