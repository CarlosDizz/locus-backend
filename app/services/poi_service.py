from __future__ import annotations

from math import cos, radians, sqrt

from sqlalchemy import Select, or_, select

from app.clients.maps_client import MapsClient
from app.clients.wikipedia_client import WikipediaClient
from app.clients.wikidata_client import WikidataClient
from app.config import settings
from app.db.models import Poi
from app.db.session import session_scope
from app.schemas.poi import POI
from app.utils.logging import get_logger
from app.utils.text import clean_text, slugify


class POIService:
    def __init__(self) -> None:
        self.maps_client = MapsClient()
        self.wikipedia_client = WikipediaClient()
        self.wikidata_client = WikidataClient()
        self.logger = get_logger(__name__)

    def _distance_km(self, lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
        lat_factor = 111.32
        lng_factor = 111.32 * max(cos(radians(lat_a)), 0.2)
        return sqrt(((lat_b - lat_a) * lat_factor) ** 2 + ((lng_b - lng_a) * lng_factor) ** 2)

    def _is_hospitality_or_service_label(self, text: str) -> bool:
        lowered = clean_text(text).lower()
        blocked = [
            "restaurante",
            "restaurant",
            "bar",
            "pub",
            "cafe",
            "café",
            "hotel",
            "hostel",
            "farmacia",
            "pharmacy",
            "supermercado",
            "taxi",
            "garden",
        ]
        return any(token in lowered for token in blocked)

    def _catalog_poi_to_runtime(self, poi: Poi) -> POI:
        description = poi.short_description or poi.long_description or ""
        summary = poi.long_description or poi.short_description or ""
        return POI(
            id=str(poi.id),
            name=poi.name,
            lat=float(poi.lat) if poi.lat is not None else 0.0,
            lng=float(poi.lng) if poi.lng is not None else 0.0,
            poi_type_code="",
            description=description,
            summary=summary,
            source_of_truth="catalog",
            is_ephemeral=False,
            google_place_id=poi.google_place_id or "",
            context_kind="catalog",
        )

    def _is_generic_query(self, query: str) -> bool:
        normalized = clean_text(query).lower()
        return normalized in {
            "",
            "lugares turisticos",
            "lugares turísticos",
            "que ver",
            "qué ver",
            "recomiendame algo",
            "recomiéndame algo",
        }

    def _search_catalog_pois(self, query: str, lat: float | None, lng: float | None, limit: int = 5) -> list[POI]:
        with session_scope() as db:
            stmt: Select[tuple[Poi]] = select(Poi).where(Poi.is_active.is_(True))
            if lat is not None and lng is not None:
                lat_delta = 10 / 111.32
                lng_delta = 10 / max(111.32 * max(cos(radians(lat)), 0.2), 1)
                stmt = stmt.where(Poi.lat.is_not(None), Poi.lng.is_not(None))
                stmt = stmt.where(Poi.lat.between(lat - lat_delta, lat + lat_delta))
                stmt = stmt.where(Poi.lng.between(lng - lng_delta, lng + lng_delta))

            if not self._is_generic_query(query):
                token = f"%{clean_text(query)}%"
                stmt = stmt.where(
                    or_(
                        Poi.name.ilike(token),
                        Poi.slug.ilike(token),
                        Poi.short_description.ilike(token),
                        Poi.long_description.ilike(token),
                    )
                )

            candidates = list(db.scalars(stmt.limit(max(limit * 6, 20))).all())

        if lat is not None and lng is not None:
            candidates = sorted(
                candidates,
                key=lambda item: self._distance_km(lat, lng, float(item.lat), float(item.lng))
                if item.lat is not None and item.lng is not None
                else 9999,
            )
        else:
            candidates = sorted(candidates, key=lambda item: item.name.lower())

        return [self._catalog_poi_to_runtime(item) for item in candidates[:limit]]

    def _find_catalog_poi(self, poi_name: str) -> Poi | None:
        cleaned_name = clean_text(poi_name)
        if not cleaned_name:
            return None
        with session_scope() as db:
            return db.scalar(
                select(Poi).where(
                    Poi.is_active.is_(True),
                    or_(Poi.name.ilike(cleaned_name), Poi.slug == slugify(cleaned_name)),
                )
            )

    def search_nearby_pois(self, query: str, lat: float | None, lng: float | None, limit: int = 5) -> list[POI]:
        catalog_results = self._search_catalog_pois(query=query, lat=lat, lng=lng, limit=limit)
        if self._is_generic_query(query):
            return catalog_results[:limit]
        if len(catalog_results) >= limit:
            return catalog_results

        raw_places = self.maps_client.search_places(query=query, lat=lat, lng=lng, limit=limit)
        external_results = [POI(**item) for item in raw_places]

        merged: list[POI] = []
        seen_names: set[str] = set()
        for item in [*catalog_results, *external_results]:
            key = slugify(item.name)
            if key in seen_names:
                continue
            seen_names.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def search_contextual_places(self, query: str, lat: float | None, lng: float | None, limit: int = 5) -> list[POI]:
        raw_places = self.maps_client.search_places(query=query, lat=lat, lng=lng, limit=limit)
        return [POI(**item) for item in raw_places]

    def search_tourism_candidates(self, query: str, lat: float | None, lng: float | None, limit: int = 5) -> list[POI]:
        results = self.search_nearby_pois(query=query, lat=lat, lng=lng, limit=limit)
        query_tokens = [token for token in slugify(query).split("-") if len(token) > 2]
        normalized: list[POI] = []
        for poi in results:
            haystack = f"{poi.name} {poi.description} {poi.summary}"
            slug = slugify(haystack)
            if self._is_hospitality_or_service_label(haystack):
                continue
            if query_tokens and not any(token in slug for token in query_tokens):
                continue
            if lat is not None and lng is not None and self._distance_km(lat, lng, poi.lat, poi.lng) > 8:
                continue
            context_kind = "catalog" if not poi.is_ephemeral else "tourism_candidate"
            normalized.append(poi.model_copy(update={"context_kind": context_kind}))
            if len(normalized) >= limit:
                break
        return normalized

    def enrich_poi(self, poi: POI) -> POI:
        if poi.summary:
            return poi
        documentation = self.get_poi_documentation(poi.name)
        poi.summary = documentation["summary"]
        return poi

    def get_poi_summary(self, poi_name: str) -> str:
        return self.get_poi_documentation(poi_name)["summary"]

    def _extract_claim_value(self, claim: dict) -> str | None:
        datavalue = (
            claim.get("mainsnak", {})
            .get("datavalue", {})
        )
        value = datavalue.get("value")
        if isinstance(value, dict) and "id" in value:
            return value["id"]
        if isinstance(value, dict) and "time" in value:
            return value["time"]
        if isinstance(value, dict) and "latitude" in value and "longitude" in value:
            return f"{value['latitude']}, {value['longitude']}"
        if isinstance(value, str):
            return value
        return None

    def _first_claim_value(self, claims: dict, property_id: str) -> str | None:
        entries = claims.get(property_id, [])
        if not entries:
            return None
        return self._extract_claim_value(entries[0])

    def get_poi_documentation(self, poi_name: str) -> dict:
        catalog_poi = self._find_catalog_poi(poi_name)
        effective_name = catalog_poi.name if catalog_poi is not None else poi_name

        if catalog_poi and catalog_poi.long_description and catalog_poi.short_description:
            summary = f"{catalog_poi.short_description} {catalog_poi.long_description}".strip()
            return {
                "poi_name": catalog_poi.name,
                "summary": summary,
                "wikidata": None,
                "facts": {},
                "sources": ["catalog"],
                "catalog_poi": {
                    "id": catalog_poi.id,
                    "name": catalog_poi.name,
                    "wikidata_id": catalog_poi.wikidata_id,
                    "wikipedia_title": catalog_poi.wikipedia_title,
                },
            }

        wiki_query = catalog_poi.wikipedia_title if catalog_poi and catalog_poi.wikipedia_title else effective_name
        wiki_summary = self.wikipedia_client.get_summary(wiki_query)
        entity_match = None
        if catalog_poi and catalog_poi.wikidata_id:
            entity_match = {"id": catalog_poi.wikidata_id, "label": catalog_poi.name, "description": catalog_poi.short_description}
        else:
            entity_match = self.wikidata_client.search_entity(effective_name)

        if not entity_match:
            summary = wiki_summary or (catalog_poi.short_description if catalog_poi else "")
            if catalog_poi and catalog_poi.long_description:
                summary = f"{summary} {catalog_poi.long_description}".strip()
            return {
                "poi_name": effective_name,
                "summary": summary,
                "wikidata": None,
                "facts": {},
                "sources": [source for source in ["catalog", "wikipedia"] if (source == "catalog" and catalog_poi) or (source == "wikipedia" and wiki_summary)],
                "catalog_poi": {
                    "id": catalog_poi.id,
                    "name": catalog_poi.name,
                    "wikidata_id": catalog_poi.wikidata_id,
                    "wikipedia_title": catalog_poi.wikipedia_title,
                } if catalog_poi else None,
            }

        entity_id = entity_match.get("id", "")
        try:
            entity = self.wikidata_client.get_entity(entity_id)
        except Exception as exc:
            self.logger.warning("Wikidata entity fetch failed for %s: %s", poi_name, exc)
            entity = {}

        claims = entity.get("claims", {})
        raw_linked_ids = [
            value
            for value in [
                self._first_claim_value(claims, "P31"),
                self._first_claim_value(claims, "P17"),
                self._first_claim_value(claims, "P131"),
            ]
            if value and value.startswith("Q")
        ]
        linked_labels = self.wikidata_client.get_entity_labels(raw_linked_ids)

        facts = {
            "instance_of": linked_labels.get(self._first_claim_value(claims, "P31") or "", ""),
            "country": linked_labels.get(self._first_claim_value(claims, "P17") or "", ""),
            "located_in": linked_labels.get(self._first_claim_value(claims, "P131") or "", ""),
            "inception": self._first_claim_value(claims, "P571") or "",
            "coordinates": self._first_claim_value(claims, "P625") or "",
        }
        facts = {key: value for key, value in facts.items() if value}

        label = (
            entity.get("labels", {})
            .get(settings.wikidata_language, {})
            .get("value")
            or entity_match.get("label", "")
            or effective_name
        )
        description = (
            entity.get("descriptions", {})
            .get(settings.wikidata_language, {})
            .get("value")
            or entity_match.get("description", "")
        )

        summary_parts = []
        if catalog_poi and catalog_poi.short_description:
            summary_parts.append(catalog_poi.short_description)
        if wiki_summary:
            summary_parts.append(wiki_summary)
        elif description:
            summary_parts.append(f"{label}: {description}.")
        if catalog_poi and catalog_poi.long_description:
            summary_parts.append(catalog_poi.long_description)

        factual_lines = []
        if facts.get("instance_of"):
            factual_lines.append(f"Tipo: {facts['instance_of']}.")
        if facts.get("located_in"):
            factual_lines.append(f"Ubicación administrativa: {facts['located_in']}.")
        if facts.get("country"):
            factual_lines.append(f"País: {facts['country']}.")
        if facts.get("inception"):
            factual_lines.append(f"Fecha registrada en Wikidata: {facts['inception']}.")
        if factual_lines:
            summary_parts.append(" ".join(factual_lines))

        sitelinks = entity.get("sitelinks", {})
        wikipedia_title = ""
        preferred_wiki = f"{settings.wikipedia_language}wiki"
        if preferred_wiki in sitelinks:
            wikipedia_title = sitelinks[preferred_wiki].get("title", "")
        elif "enwiki" in sitelinks:
            wikipedia_title = sitelinks["enwiki"].get("title", "")

        return {
            "poi_name": label,
            "summary": " ".join(part.strip() for part in summary_parts if part).strip(),
            "wikidata": {
                "id": entity_id,
                "label": label,
                "description": description,
                "wikipedia_title": wikipedia_title,
                "url": f"{self.wikidata_client.base_url}/wiki/{entity_id}",
            },
            "facts": facts,
            "sources": [
                source
                for source in ["catalog", "wikidata", "wikipedia"]
                if (source == "catalog" and catalog_poi)
                or source == "wikidata"
                or (source == "wikipedia" and wiki_summary)
            ],
            "catalog_poi": {
                "id": catalog_poi.id,
                "name": catalog_poi.name,
                "wikidata_id": catalog_poi.wikidata_id,
                "wikipedia_title": catalog_poi.wikipedia_title,
            } if catalog_poi else None,
        }


poi_service = POIService()
