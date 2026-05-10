from __future__ import annotations

import json
import math
import re
import threading
import time
from decimal import Decimal

import requests
from requests import RequestException
from sqlalchemy import Select, or_, select

from app.clients.overpass_client import OverpassClient
from app.clients.openai_client import OpenAIClient, OpenAIClientError
from app.clients.wikidata_client import WikidataClient
from app.db.models import City, Poi, PoiType
from app.db.session import session_scope
from app.schemas.catalog import CityCreateRequest, CityResponse, PoiCreateRequest, PoiResponse, PoiTypeResponse, PoiUpdateRequest
from app.utils.logging import get_logger
from app.utils.text import clean_text, slugify


class CatalogError(RuntimeError):
    pass


class CatalogService:
    MIN_BOOTSTRAP_POI_COUNT = 12
    SPARSE_CATALOG_RETRY_POI_COUNT = 6

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self._enrichment_lock = threading.Lock()
        self._active_enrichment_jobs: set[int] = set()
        self.openai = OpenAIClient()
        self.wikidata = WikidataClient()
        self.overpass = OverpassClient()
        self.poi_type_map = {
            "tourist attraction": "monument",
            "atraccion turistica": "monument",
            "museum": "museum",
            "museo": "museum",
            "art museum": "museum",
            "church building": "church",
            "iglesia": "church",
            "church": "church",
            "cathedral": "church",
            "catedral": "church",
            "basilica": "church",
            "basilica menor": "church",
            "square": "square",
            "plaza": "square",
            "monument": "monument",
            "monumento": "monument",
            "palace": "building",
            "palacio": "building",
            "alcazar": "monument",
            "alcázar": "monument",
            "synagogue": "church",
            "sinagoga": "church",
            "monasterio": "church",
            "monastery": "church",
            "convento": "church",
            "teatro": "building",
            "archaeological site": "archaeological_site",
            "yacimiento arqueologico": "archaeological_site",
            "yacimiento arqueológico": "archaeological_site",
            "anfiteatro romano": "archaeological_site",
            "anfiteatro": "archaeological_site",
            "foro romano": "archaeological_site",
            "bridge": "building",
            "building": "building",
        }
        self.overpass_tag_type_map = {
            "tourism:museum": "museum",
            "tourism:gallery": "museum",
            "tourism:attraction": "monument",
            "historic:monument": "monument",
            "historic:memorial": "monument",
            "historic:castle": "monument",
            "historic:archaeological_site": "archaeological_site",
            "amenity:place_of_worship": "church",
            "building:cathedral": "church",
            "building:church": "church",
            "building:synagogue": "church",
            "place:square": "square",
        }
        self.tourism_type_scores = {
            "archaeological_site": 120,
            "monument": 115,
            "museum": 110,
            "church": 95,
            "square": 85,
            "building": 55,
        }
        self.tourism_positive_terms = {
            "catedral": 60,
            "cathedral": 60,
            "museo": 55,
            "museum": 55,
            "plaza": 40,
            "square": 40,
            "palacio": 40,
            "palace": 40,
            "castillo": 55,
            "castle": 55,
            "puerta": 35,
            "gate": 35,
            "iglesia": 35,
            "church": 35,
            "basílica": 45,
            "basilica": 45,
            "monumento": 50,
            "monument": 50,
            "anfiteatro": 70,
            "teatro": 55,
            "foro": 60,
            "arqueológico": 65,
            "arqueologico": 65,
            "archaeological": 65,
            "histórico": 25,
            "historico": 25,
            "histórico-artístico": 35,
            "patrimonio": 35,
            "tourist": 20,
            "turístico": 20,
            "turistico": 20,
        }
        self.tourism_negative_terms = {
            "justicia": -140,
            "judicial": -140,
            "tribunal": -140,
            "juzgado": -140,
            "court": -140,
            "administrativo": -120,
            "administrative": -120,
            "gobierno": -100,
            "government": -100,
            "oficina": -100,
            "office": -100,
            "hospital": -120,
            "estación": -90,
            "estacion": -90,
            "station": -90,
            "aeropuerto": -120,
            "airport": -120,
            "universidad": -60,
            "university": -60,
            "colegio": -60,
            "school": -60,
            "prisón": -160,
            "prision": -160,
            "prison": -160,
            "complejo judicial": -180,
            "fiscalía": -140,
            "fiscalia": -140,
            "sede": -45,
            "campus": -60,
            "policía": -140,
            "policia": -140,
            "police": -140,
            "barrio": -180,
            "neighborhood": -180,
            "district": -150,
            "estadio": -130,
            "stadium": -130,
            "fútbol": -120,
            "futbol": -120,
            "football": -120,
        }
        self.expected_type_terms = {
            "church": {
                "positive": {
                    "church", "iglesia", "cathedral", "catedral", "basilica", "basílica",
                    "synagogue", "sinagoga", "mosque", "mezquita", "monastery", "monasterio",
                    "convent", "convento", "temple", "templo", "parish", "parroquia", "ermita", "ermitaño",
                },
                "negative": {
                    "library", "biblioteca", "museum", "museo", "school", "colegio", "hospital",
                    "stadium", "estadio", "office", "oficina", "hotel", "restaurant",
                },
            },
            "museum": {
                "positive": {
                    "museum", "museo", "gallery", "galería", "exhibition", "colección", "collection",
                    "visitor centre", "centro de interpretación",
                },
                "negative": {
                    "church", "iglesia", "cathedral", "catedral", "synagogue", "sinagoga",
                    "bridge", "puente", "gate", "puerta", "square", "plaza",
                },
            },
            "monument": {
                "positive": {
                    "monument", "monumento", "gate", "puerta", "bridge", "puente", "castle", "castillo",
                    "fortress", "fortaleza", "tower", "torre", "wall", "muralla", "alcázar", "alcazar",
                },
                "negative": {
                    "library", "biblioteca", "hospital", "school", "colegio", "stadium", "estadio",
                },
            },
            "square": {
                "positive": {"square", "plaza", "piazza", "place"},
                "negative": {"church", "iglesia", "museum", "museo", "hospital", "biblioteca"},
            },
            "building": {
                "positive": {
                    "building", "edificio", "palace", "palacio", "hospital", "college", "colegio",
                    "castle", "castillo", "alcázar", "alcazar",
                },
                "negative": set(),
            },
            "archaeological_site": {
                "positive": {
                    "archaeological", "arqueológico", "arqueologico", "ruins", "ruinas",
                    "roman", "romano", "circus", "circo", "thermae", "termas", "forum", "foro",
                    "site", "yacimiento",
                },
                "negative": {"library", "biblioteca", "office", "oficina", "hospital"},
            },
        }

    def _city_to_schema(self, city: City) -> CityResponse:
        return CityResponse(
            id=city.id,
            slug=city.slug,
            name=city.name,
            country_code=city.country_code,
            lat=float(city.lat) if city.lat is not None else None,
            lng=float(city.lng) if city.lng is not None else None,
            source=city.source,
            created_at=city.created_at,
        )

    def _candidate_text(self, candidate: dict) -> str:
        return " ".join(
            [
                clean_text(candidate.get("label", "")),
                clean_text(candidate.get("description", "")),
            ]
        ).lower()

    def _score_city_candidate(self, query: str, candidate: dict, country_code: str = "") -> int:
        normalized_query = clean_text(query).lower()
        label = clean_text(candidate.get("label", "")).lower()
        description = clean_text(candidate.get("description", "")).lower()
        text = f"{label} {description}".strip()
        score = 0

        if label == normalized_query:
            score += 120
        elif normalized_query and normalized_query in label:
            score += 40

        if any(token in description for token in ["capital", "city", "ciudad", "comuna", "municipio", "municipality"]):
            score += 80
        if any(token in description for token in ["football", "fútbol", "soccer", "club", "team", "f.c."]):
            score -= 200
        if any(token in text for token in ["roman empire", "película", "album", "song", "novel"]):
            score -= 80
        if country_code:
            country_terms = {
                "IT": ["italy", "italia"],
                "ES": ["spain", "españa"],
                "FR": ["france", "francia"],
                "GB": ["united kingdom", "england", "reino unido"],
                "US": ["united states", "usa", "estados unidos"],
            }.get(country_code.upper(), [])
            if country_terms and any(term in description for term in country_terms):
                score += 20
        return score

    def _resolve_city_entity_id(self, city: City) -> str | None:
        searches = [
            *self.wikidata.search_entities(city.name, limit=8),
            *self.wikidata.search_entities(f"{city.name} city", limit=8),
            *self.wikidata.search_entities(city.slug.replace("-", " "), limit=8),
        ]
        best_id = ""
        best_score = -9999
        seen_ids: set[str] = set()
        for candidate in searches:
            candidate_id = candidate.get("id", "")
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            score = self._score_city_candidate(city.name, candidate, city.country_code)
            if score > best_score:
                best_id = candidate_id
                best_score = score
        return best_id or None

    def _country_terms(self, country_code: str) -> list[str]:
        return {
            "IT": ["italy", "italia"],
            "ES": ["spain", "españa", "castilla-la mancha", "castilla la mancha"],
            "FR": ["france", "francia"],
            "GB": ["united kingdom", "england", "reino unido"],
            "US": ["united states", "usa", "estados unidos"],
        }.get((country_code or "").upper(), [])

    def _poi_type_by_code(self, db, code: str | None) -> PoiType | None:
        if not code:
            return None
        return db.scalar(select(PoiType).where(PoiType.code == code))

    def _poi_to_schema(self, poi: Poi, poi_type: PoiType | None = None) -> PoiResponse:
        return PoiResponse(
            id=poi.id,
            city_id=poi.city_id,
            poi_type_id=poi.poi_type_id,
            poi_type_code=poi_type.code if poi_type else None,
            poi_type_name=poi_type.name if poi_type else None,
            slug=poi.slug,
            name=poi.name,
            lat=float(poi.lat) if poi.lat is not None else None,
            lng=float(poi.lng) if poi.lng is not None else None,
            short_description=poi.short_description,
            long_description=poi.long_description,
            source_of_truth=poi.source_of_truth,
            wikidata_id=poi.wikidata_id,
            wikipedia_title=poi.wikipedia_title,
            google_place_id=poi.google_place_id,
            is_active=poi.is_active,
            metadata=poi.metadata_json or {},
            created_at=poi.created_at,
            updated_at=poi.updated_at,
        )

    def list_poi_types(self) -> list[PoiTypeResponse]:
        with session_scope() as db:
            poi_types = list(db.scalars(select(PoiType).order_by(PoiType.name)).all())
            return [
                PoiTypeResponse(id=item.id, code=item.code, name=item.name, description=item.description)
                for item in poi_types
            ]

    def get_poi(self, poi_id: int) -> PoiResponse:
        with session_scope() as db:
            poi = db.get(Poi, poi_id)
            if poi is None:
                raise CatalogError("POI no encontrado")
            poi_type = db.get(PoiType, poi.poi_type_id) if poi.poi_type_id else None
            return self._poi_to_schema(poi, poi_type)

    def list_cities(self, q: str = "", limit: int = 100) -> list[CityResponse]:
        with session_scope() as db:
            stmt: Select[tuple[City]] = select(City).order_by(City.name).limit(limit)
            if q.strip():
                token = f"%{q.strip()}%"
                stmt = select(City).where(or_(City.name.ilike(token), City.slug.ilike(token))).order_by(City.name).limit(limit)
            cities = list(db.scalars(stmt).all())
            return [self._city_to_schema(city) for city in cities]

    def create_city(self, payload: CityCreateRequest) -> CityResponse:
        with session_scope() as db:
            slug = payload.slug or slugify(payload.name)
            existing = db.scalar(select(City).where(City.slug == slug))
            if existing is not None:
                raise CatalogError("Ya existe una ciudad con ese slug")
            city = City(
                slug=slug,
                name=clean_text(payload.name),
                country_code=payload.country_code.upper(),
                lat=Decimal(str(payload.lat)) if payload.lat is not None else None,
                lng=Decimal(str(payload.lng)) if payload.lng is not None else None,
                source=payload.source,
            )
            db.add(city)
            db.flush()
            return self._city_to_schema(city)

    def bootstrap_city(self, name: str, country_code: str = "") -> CityResponse:
        query = clean_text(name)
        if not query:
            raise CatalogError("El nombre de ciudad es obligatorio")
        searches = [
            *self.wikidata.search_entities(query, limit=8),
            *self.wikidata.search_entities(f"{query} city", limit=8),
            *self.wikidata.search_entities(f"{query} capital", limit=8),
        ]
        deduped: list[dict] = []
        seen_ids: set[str] = set()
        for candidate in searches:
            candidate_id = candidate.get("id", "")
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            deduped.append(candidate)

        if not deduped:
            raise CatalogError("No he podido resolver la ciudad en Wikidata")

        best = max(deduped, key=lambda item: self._score_city_candidate(query, item, country_code))
        if self._score_city_candidate(query, best, country_code) < 50:
            raise CatalogError("No he podido resolver con confianza una ciudad válida en Wikidata")

        entity = self.wikidata.get_entity(best["id"])
        coords_claim = (entity.get("claims", {}).get("P625") or [])
        lat = None
        lng = None
        if coords_claim:
            value = coords_claim[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
            lat = value.get("latitude")
            lng = value.get("longitude")
        city_name = (
            entity.get("labels", {}).get("es", {}).get("value")
            or best.get("label")
            or query
        )
        return self.create_city(
            CityCreateRequest(
                name=city_name,
                slug=slugify(city_name),
                country_code=country_code.upper(),
                lat=lat,
                lng=lng,
                source="wikidata",
            )
        )

    def _resolve_city_name_from_coords(self, lat: float, lng: float) -> tuple[str, str]:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "format": "jsonv2",
                    "lat": lat,
                    "lon": lng,
                    "zoom": 10,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "LocusBackend/1.0"},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise CatalogError(f"No he podido resolver la ciudad desde la ubicación: {exc}") from exc

        address = payload.get("address", {}) or {}
        city_name = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or ""
        )
        country_code = str(address.get("country_code") or "").upper()

        city_name = clean_text(city_name)
        if not city_name:
            raise CatalogError("No he podido deducir una ciudad válida desde tu ubicación")
        return city_name, country_code

    def bootstrap_city_from_location(
        self,
        *,
        lat: float,
        lng: float,
        radius_km: float = 8.0,
        limit: int = 80,
        use_ai_candidates: bool = True,
    ) -> tuple[CityResponse, int, int, int, dict, list[PoiResponse]]:
        city_name, country_code = self._resolve_city_name_from_coords(lat, lng)
        slug = slugify(city_name)

        with session_scope() as db:
            existing = db.scalar(select(City).where(City.slug == slug))
            city = self._city_to_schema(existing) if existing is not None else None

        if city is None:
            city = self.bootstrap_city(city_name, country_code)

        pois = self.list_pois(city_id=city.id, limit=limit)
        existing_poi_count = len(pois)
        if existing_poi_count >= self.MIN_BOOTSTRAP_POI_COUNT:
            return city, 0, 0, 0, {"source": "existing_catalog", "existing_poi_count": existing_poi_count}, pois

        imported_count, updated_count, skipped_count, stats, pois, _city_name = self.import_city_pois(
            city_id=city.id,
            radius_km=radius_km,
            limit=limit,
            use_ai_candidates=use_ai_candidates,
        )
        used_ai_fallback = False
        if (
            not use_ai_candidates
            and len(pois) < self.SPARSE_CATALOG_RETRY_POI_COUNT
        ):
            ai_imported, ai_updated, ai_skipped, ai_stats, ai_pois, _city_name = self.import_city_pois(
                city_id=city.id,
                radius_km=radius_km,
                limit=limit,
                use_ai_candidates=True,
            )
            imported_count += ai_imported
            updated_count += ai_updated
            skipped_count += ai_skipped
            if ai_pois:
                pois = ai_pois
            stats = {
                "source": "catalog_sources_then_ai_fallback",
                "existing_poi_count": existing_poi_count,
                "catalog_attempt": stats,
                "ai_fallback": ai_stats,
            }
            used_ai_fallback = True

        if not used_ai_fallback:
            stats["source"] = "existing_catalog_reimport" if existing_poi_count else "fresh_import"
            stats["existing_poi_count"] = existing_poi_count
        if use_ai_candidates or used_ai_fallback:
            self.start_pending_enrichment(city.id, min(limit, 150))
        return city, imported_count, updated_count, skipped_count, stats, pois

    def list_pois(
        self,
        *,
        city_id: int | None = None,
        poi_type_code: str | None = None,
        q: str = "",
        limit: int = 200,
    ) -> list[PoiResponse]:
        with session_scope() as db:
            stmt = select(Poi, PoiType).outerjoin(PoiType, Poi.poi_type_id == PoiType.id).order_by(Poi.name).limit(limit)
            if city_id is not None:
                stmt = stmt.where(Poi.city_id == city_id)
            if poi_type_code:
                stmt = stmt.where(PoiType.code == poi_type_code)
            if q.strip():
                token = f"%{q.strip()}%"
                stmt = stmt.where(or_(Poi.name.ilike(token), Poi.slug.ilike(token), Poi.short_description.ilike(token)))
            rows = db.execute(stmt).all()
            return [self._poi_to_schema(poi, poi_type) for poi, poi_type in rows]

    def create_poi(self, payload: PoiCreateRequest) -> PoiResponse:
        with session_scope() as db:
            poi_type = db.get(PoiType, payload.poi_type_id) if payload.poi_type_id else self._poi_type_by_code(db, payload.poi_type_code)
            slug = payload.slug or slugify(payload.name)
            poi = Poi(
                city_id=payload.city_id,
                poi_type_id=poi_type.id if poi_type else None,
                slug=slug,
                name=clean_text(payload.name),
                lat=Decimal(str(payload.lat)) if payload.lat is not None else None,
                lng=Decimal(str(payload.lng)) if payload.lng is not None else None,
                short_description=clean_text(payload.short_description),
                long_description=(payload.long_description or "").strip(),
                source_of_truth=payload.source_of_truth,
                wikidata_id=payload.wikidata_id,
                wikipedia_title=payload.wikipedia_title,
                google_place_id=payload.google_place_id,
                is_active=payload.is_active,
                metadata_json=payload.metadata,
            )
            db.add(poi)
            db.flush()
            return self._poi_to_schema(poi, poi_type)

    def update_poi(self, poi_id: int, payload: PoiUpdateRequest) -> PoiResponse:
        with session_scope() as db:
            poi = db.get(Poi, poi_id)
            if poi is None:
                raise CatalogError("POI no encontrado")
            poi_type = db.get(PoiType, payload.poi_type_id) if payload.poi_type_id else self._poi_type_by_code(db, payload.poi_type_code)
            if payload.city_id is not None:
                poi.city_id = payload.city_id
            if poi_type is not None:
                poi.poi_type_id = poi_type.id
            if payload.slug is not None:
                poi.slug = payload.slug
            if payload.name is not None:
                poi.name = clean_text(payload.name)
            if payload.lat is not None:
                poi.lat = Decimal(str(payload.lat))
            if payload.lng is not None:
                poi.lng = Decimal(str(payload.lng))
            if payload.short_description is not None:
                poi.short_description = clean_text(payload.short_description)
            if payload.long_description is not None:
                poi.long_description = payload.long_description.strip()
            if payload.source_of_truth is not None:
                poi.source_of_truth = payload.source_of_truth
            if payload.wikidata_id is not None:
                poi.wikidata_id = payload.wikidata_id
            if payload.wikipedia_title is not None:
                poi.wikipedia_title = payload.wikipedia_title
            if payload.google_place_id is not None:
                poi.google_place_id = payload.google_place_id
            if payload.is_active is not None:
                poi.is_active = payload.is_active
            if payload.metadata:
                current = dict(poi.metadata_json or {})
                current.update(payload.metadata)
                poi.metadata_json = current
            db.flush()
            poi_type = poi_type or (db.get(PoiType, poi.poi_type_id) if poi.poi_type_id else None)
            return self._poi_to_schema(poi, poi_type)

    def _wkt_point_to_coords(self, wkt: str) -> tuple[float | None, float | None]:
        match = re.match(r"Point\(([-0-9.]+)\s+([-0-9.]+)\)", wkt or "")
        if not match:
            return None, None
        lng = float(match.group(1))
        lat = float(match.group(2))
        return lat, lng

    def _map_poi_type_code(self, name: str, type_label: str, fallback_description: str) -> str:
        label = (type_label or "").strip().lower()
        description = (fallback_description or "").strip().lower()
        title = (name or "").strip().lower()
        combined = " ".join(part for part in [title, label, description] if part)
        for token, code in self.poi_type_map.items():
            if token in combined:
                return code
        return "building"

    def _extract_response_text(self, response: dict) -> str:
        output_text = response.get("output_text")
        if output_text:
            return output_text.strip()
        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    texts.append(content["text"])
        return "\n".join(texts).strip()

    def _extract_json_object(self, text: str) -> dict:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CatalogError("La respuesta del modelo no contiene un JSON válido")
        return json.loads(raw[start : end + 1])

    def _build_ai_candidate_instructions(self, city: City, limit: int) -> str:
        poi_types = ", ".join(
            f"{code} ({label})"
            for code, label in [
                ("monument", "monumento o hito histórico"),
                ("museum", "museo o galería"),
                ("church", "catedral, iglesia, sinagoga, monasterio o templo"),
                ("square", "plaza o espacio cívico emblemático"),
                ("building", "edificio visitable relevante"),
                ("archaeological_site", "yacimiento o ruina arqueológica"),
            ]
        )
        return (
            "Devuelve candidatos de puntos de interés turísticos para una ciudad en JSON puro. "
            "No uses markdown, no expliques nada y no rellenes la lista si la ciudad tiene pocos sitios claros. "
            f"El límite superior es {limit}, pero puedes devolver menos. "
            "Si la ciudad es muy turística, intenta acercarte a 20-40 candidatos distintos y específicos. "
            "Prioriza lugares famosos, visitables o claramente reconocibles por viajeros. "
            "Incluye nombres alternativos útiles para resolver el sitio en Wikidata: nombre oficial, variante local o forma histórica breve. "
            "Si conoces con suficiente confianza una ubicación utilizable, devuelve lat y lng. "
            "Si no estás seguro de las coordenadas, devuelve null y no inventes. "
            "Si conoces una dirección o referencia postal razonable, devuélvela en formatted_address; si no, null. "
            "Incluye location_hint con una referencia geográfica breve y útil para situar el lugar, por ejemplo barrio, zona o municipio. "
            "Incluye short_description con una sola frase breve y neutra sobre por qué es relevante. "
            "Evita barrios, conjuntos demasiado genéricos, juzgados, hospitales, oficinas, estaciones, estadios y relleno dudoso. "
            f"Usa solo estos poi_type_code: {poi_types}. "
            "Devuelve este shape exacto: "
            '{"city":"nombre","items":[{"name":"...","poi_type_code":"...","aliases":["..."],"formatted_address":"... o null","location_hint":"... o null","short_description":"...","lat":0.0 o null,"lng":0.0 o null}]}.'
        )

    def _normalize_optional_text(self, value: object, *, limit: int = 255) -> str:
        if value is None:
            return ""
        text = clean_text(str(value))
        if not text:
            return ""
        return text[:limit]

    def _normalize_optional_float(self, value: object) -> float | None:
        if value is None or value == "":
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number

    def _generate_ai_candidates(self, city: City, limit: int) -> list[dict]:
        if not self.openai.is_configured():
            return []
        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Ciudad: {city.name}\n"
                            f"País: {city.country_code or '(sin dato)'}\n"
                            f"Límite máximo: {limit}\n"
                            "Devuelve solo el JSON."
                        ),
                    }
                ],
            }
        ]
        try:
            response = self.openai.create_response(
                model=self.openai.chat_model(),
                instructions=self._build_ai_candidate_instructions(city, limit),
                input_items=input_items,
                max_output_tokens=min(12000, max(2000, 400 + (limit * 90))),
                tool_choice="none",
                extra_payload={
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "city_poi_candidates",
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "city": {"type": "string"},
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "name": {"type": "string"},
                                                "poi_type_code": {
                                                    "type": "string",
                                                    "enum": [
                                                        "monument",
                                                        "museum",
                                                        "church",
                                                        "square",
                                                        "building",
                                                        "archaeological_site",
                                                    ],
                                                },
                                                "aliases": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "formatted_address": {
                                                    "type": ["string", "null"],
                                                },
                                                "location_hint": {
                                                    "type": ["string", "null"],
                                                },
                                                "short_description": {
                                                    "type": "string",
                                                },
                                                "lat": {
                                                    "type": ["number", "null"],
                                                },
                                                "lng": {
                                                    "type": ["number", "null"],
                                                },
                                            },
                                            "required": [
                                                "name",
                                                "poi_type_code",
                                                "aliases",
                                                "formatted_address",
                                                "location_hint",
                                                "short_description",
                                                "lat",
                                                "lng",
                                            ],
                                        },
                                    },
                                },
                                "required": ["city", "items"],
                            },
                        }
                    }
                },
            )
        except OpenAIClientError as exc:
            raise CatalogError(f"No he podido generar candidatos con OpenAI: {exc}") from exc

        payload = self._extract_json_object(self._extract_response_text(response))
        items = payload.get("items", [])
        candidates: list[dict] = []
        for item in items[:limit]:
            name = clean_text(item.get("name", ""))
            poi_type_code = clean_text(item.get("poi_type_code", "")).lower()
            aliases = [clean_text(alias) for alias in item.get("aliases", []) if clean_text(alias)]
            formatted_address = self._normalize_optional_text(item.get("formatted_address"), limit=255)
            location_hint = self._normalize_optional_text(item.get("location_hint"), limit=255)
            short_description = self._normalize_optional_text(item.get("short_description"), limit=500)
            lat = self._normalize_optional_float(item.get("lat"))
            lng = self._normalize_optional_float(item.get("lng"))
            if lat is not None and not (-90 <= lat <= 90):
                lat = None
            if lng is not None and not (-180 <= lng <= 180):
                lng = None
            if (lat is None) != (lng is None):
                lat = None
                lng = None
            if not name or poi_type_code not in {"monument", "museum", "church", "square", "building", "archaeological_site"}:
                continue
            candidates.append(
                {
                    "name": name,
                    "poi_type_code": poi_type_code,
                    "aliases": aliases[:5],
                    "formatted_address": formatted_address,
                    "location_hint": location_hint,
                    "short_description": short_description,
                    "lat": lat,
                    "lng": lng,
                }
            )
        return candidates

    def _candidate_metadata(self, candidate: dict, *, seed_rank: int, status: str = "pending_wikidata") -> dict:
        return {
            "candidate_aliases": candidate.get("aliases", [])[:5],
            "candidate_name": candidate["name"],
            "candidate_type_code": candidate["poi_type_code"],
            "import_status": status,
            "seed_model": self.openai.chat_model(),
            "seed_rank": seed_rank,
            "seed_source": "gpt_candidate",
            "resolution_attempts": 0,
            "featured": False,
            "import_tier": "pending",
            "catalog_source": "gpt_seed",
            "formatted_address": candidate.get("formatted_address") or "",
            "location_hint": candidate.get("location_hint") or "",
            "seed_short_description": candidate.get("short_description") or "",
            "seed_lat": candidate.get("lat"),
            "seed_lng": candidate.get("lng"),
        }

    def _build_pending_candidate(self, poi: Poi) -> dict:
        metadata = dict(poi.metadata_json or {})
        aliases = metadata.get("candidate_aliases") or []
        candidate_type_code = clean_text(metadata.get("candidate_type_code") or "") or "building"
        return {
            "name": poi.name,
            "poi_type_code": candidate_type_code,
            "aliases": [clean_text(alias) for alias in aliases if clean_text(alias)],
            "formatted_address": clean_text(metadata.get("formatted_address") or ""),
            "location_hint": clean_text(metadata.get("location_hint") or ""),
            "short_description": clean_text(metadata.get("seed_short_description") or poi.short_description),
            "lat": metadata.get("seed_lat"),
            "lng": metadata.get("seed_lng"),
        }

    def _upsert_ai_seed_candidates(
        self,
        db,
        city: City,
        type_lookup: dict[str, PoiType],
        ai_candidates: list[dict],
        limit: int,
    ) -> tuple[int, int, int, list[PoiResponse], dict]:
        imported_count = 0
        updated_count = 0
        skipped_count = 0
        rows: list[PoiResponse] = []
        seen_names: set[str] = set()

        for rank, candidate in enumerate(ai_candidates[:limit], start=1):
            dedupe_key = slugify(candidate["name"])
            if dedupe_key in seen_names:
                skipped_count += 1
                continue
            seen_names.add(dedupe_key)

            poi_type = type_lookup.get(candidate["poi_type_code"])
            existing = db.scalar(
                select(Poi)
                .where(Poi.city_id == city.id)
                .where(or_(Poi.slug == dedupe_key, Poi.name == candidate["name"]))
            )
            metadata = self._candidate_metadata(candidate, seed_rank=rank)
            has_seed_coords = candidate.get("lat") is not None and candidate.get("lng") is not None
            if has_seed_coords:
                metadata["import_status"] = "seeded_gpt_coords"
                metadata["import_tier"] = "featured"
                metadata["featured"] = True
                metadata["coord_source"] = "gpt_seed"
                metadata["coord_confidence"] = "provisional"
            if existing is None:
                poi = Poi(
                    city_id=city.id,
                    poi_type_id=poi_type.id if poi_type else None,
                    slug=dedupe_key,
                    name=candidate["name"],
                    lat=Decimal(str(candidate["lat"])) if has_seed_coords else None,
                    lng=Decimal(str(candidate["lng"])) if has_seed_coords else None,
                    short_description=(
                        candidate.get("short_description")
                        or ("Ubicación provisional generada por GPT." if has_seed_coords else "Pendiente de resolver ubicación exacta.")
                    ),
                    long_description="",
                    source_of_truth="gpt_seed",
                    wikidata_id="",
                    wikipedia_title="",
                    is_active=True,
                    metadata_json=metadata,
                )
                db.add(poi)
                db.flush()
                imported_count += 1
                rows.append(self._poi_to_schema(poi, poi_type))
                continue

            current_meta = dict(existing.metadata_json or {})
            current_meta.update(metadata)
            if existing.lat is not None and existing.lng is not None:
                current_meta["import_status"] = "resolved"
                current_meta["import_tier"] = current_meta.get("import_tier", "featured")
            elif has_seed_coords:
                existing.lat = Decimal(str(candidate["lat"]))
                existing.lng = Decimal(str(candidate["lng"]))
                current_meta["import_status"] = "seeded_gpt_coords"
                current_meta["import_tier"] = current_meta.get("import_tier", "featured")
                current_meta["featured"] = True
                current_meta["coord_source"] = "gpt_seed"
                current_meta["coord_confidence"] = "provisional"
            existing.metadata_json = current_meta
            existing.poi_type_id = poi_type.id if poi_type else existing.poi_type_id
            if candidate.get("short_description"):
                existing.short_description = candidate["short_description"]
            elif not existing.short_description:
                existing.short_description = "Ubicación provisional generada por GPT." if has_seed_coords else "Pendiente de resolver ubicación exacta."
            if existing.source_of_truth in {"wikidata", "overpass"} and existing.lat is not None and existing.lng is not None:
                existing.source_of_truth = existing.source_of_truth
            else:
                existing.source_of_truth = "gpt_seed"
            updated_count += 1
            rows.append(self._poi_to_schema(existing, poi_type or (db.get(PoiType, existing.poi_type_id) if existing.poi_type_id else None)))

        pending_count = sum(1 for row in rows if (row.metadata or {}).get("import_status") in {"pending_wikidata", "retry_wikidata", "rate_limited_retry"})
        stats = {
            "mode": "ai_seed_async",
            "ai": {
                "proposed_count": len(ai_candidates),
                "seeded_count": imported_count + updated_count,
                "duplicate_count": skipped_count,
                "seed_limit": limit,
                "seeded_with_coords_count": sum(
                    1 for candidate in ai_candidates[:limit] if candidate.get("lat") is not None and candidate.get("lng") is not None
                ),
                "seeded_with_address_count": sum(1 for candidate in ai_candidates[:limit] if candidate.get("formatted_address")),
            },
            "sources": {
                "wikidata_rows": 0,
                "overpass_rows": 0,
            },
            "ranked_candidate_count": len(rows),
            "selected_candidate_count": len(rows),
            "pending_count": pending_count,
            "enrichment_status": "queued" if pending_count else "not_needed",
        }
        return imported_count, updated_count, skipped_count, rows, stats

    def _score_wikidata_resolution(self, city: City, candidate_name: str, result: dict, aliases: list[str] | None = None) -> int:
        label = clean_text(result.get("label", "")).lower()
        description = clean_text(result.get("description", "")).lower()
        target = clean_text(candidate_name).lower()
        alias_tokens = [clean_text(alias).lower() for alias in aliases or [] if clean_text(alias)]
        score = 0
        if label == target:
            score += 120
        elif target and target in label:
            score += 60
        elif label and label in target:
            score += 45
        if any(label == alias for alias in alias_tokens):
            score += 90
        elif any(alias and alias in label for alias in alias_tokens):
            score += 55
        target_words = {token for token in re.split(r"\W+", target) if len(token) > 2}
        label_words = {token for token in re.split(r"\W+", label) if len(token) > 2}
        overlap = len(target_words & label_words)
        score += overlap * 12
        if clean_text(city.name).lower() in description:
            score += 80
        for token in self._country_terms(city.country_code):
            if token in description:
                score += 20
        match = result.get("match") or {}
        if isinstance(match, dict):
            if match.get("type") == "label":
                score += 10
            if match.get("type") == "alias":
                score += 18
        if any(token in description for token in ["district", "barrio", "neighborhood", "stadium", "football", "hospital", "court"]):
            score -= 120
        return score

    def _score_expected_type_compatibility(self, candidate_type_code: str, label: str, description: str) -> int:
        rules = self.expected_type_terms.get(candidate_type_code, {})
        if not rules:
            return 0
        text = f"{clean_text(label).lower()} {clean_text(description).lower()}".strip()
        score = 0
        positives = rules.get("positive", set())
        negatives = rules.get("negative", set())

        if any(term in text for term in positives):
            score += 90
        if any(term in text for term in negatives):
            score -= 180
        return score

    def _extract_entity_coords(self, entity: dict) -> tuple[float | None, float | None]:
        coords_claim = (entity.get("claims", {}).get("P625") or [])
        if not coords_claim:
            return None, None
        value = coords_claim[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
        return value.get("latitude"), value.get("longitude")

    def _extract_claim_entity_ids(self, entity: dict, property_id: str) -> set[str]:
        values: set[str] = set()
        for claim in entity.get("claims", {}).get(property_id, []) or []:
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            entity_id = value.get("id")
            if entity_id:
                values.add(entity_id)
        return values

    def _distance_km(
        self,
        lat_a: float | None,
        lng_a: float | None,
        lat_b: float | None,
        lng_b: float | None,
    ) -> float | None:
        if None in {lat_a, lng_a, lat_b, lng_b}:
            return None
        lat1 = math.radians(float(lat_a))
        lng1 = math.radians(float(lng_a))
        lat2 = math.radians(float(lat_b))
        lng2 = math.radians(float(lng_b))
        d_lat = lat2 - lat1
        d_lng = lng2 - lng1
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371.0 * c

    def _score_ai_entity_candidate(
        self,
        city: City,
        city_entity_id: str | None,
        candidate: dict,
        result: dict,
        entity: dict,
    ) -> tuple[int, float | None, str]:
        description = (
            entity.get("descriptions", {}).get("es", {}).get("value")
            or result.get("description", "")
            or ""
        )
        label = clean_text(result.get("label", ""))
        lat, lng = self._extract_entity_coords(entity)
        if lat is None or lng is None:
            return -9999, None, "missing_coordinates"

        score = self._score_wikidata_resolution(city, candidate["name"], result, candidate.get("aliases", []))
        distance_km = self._distance_km(city.lat, city.lng, lat, lng)
        if distance_km is not None:
            if distance_km <= 2:
                score += 120
            elif distance_km <= 5:
                score += 95
            elif distance_km <= 12:
                score += 65
            elif distance_km <= 25:
                score += 35
            elif distance_km <= 60:
                score += 10
            else:
                score -= 140

        context_ids = (
            self._extract_claim_entity_ids(entity, "P131")
            | self._extract_claim_entity_ids(entity, "P276")
            | self._extract_claim_entity_ids(entity, "P361")
        )
        if city_entity_id and city_entity_id in context_ids:
            score += 130

        score += self._score_expected_type_compatibility(candidate["poi_type_code"], label, description)

        guessed_type = self._map_poi_type_code(label, "", description)
        if guessed_type == candidate["poi_type_code"]:
            score += 30

        sitelinks = entity.get("sitelinks", {})
        score += min(len(sitelinks), 40)
        if any(token in clean_text(description).lower() for token in ["district", "barrio", "neighborhood", "stadium", "football", "hospital", "court"]):
            score -= 140
        if distance_km is not None and distance_km > 80:
            return score, distance_km, "too_far"
        if score < 95:
            return score, distance_km, "low_confidence"
        return score, distance_km, "resolved"

    def _build_ai_search_terms(self, city: City, candidate: dict) -> list[str]:
        city_name = clean_text(city.name)
        terms = [
            candidate["name"],
            f'{candidate["name"]} {city_name}',
            f'{candidate["name"]} de {city_name}',
            f'{candidate["name"]} {city.country_code}'.strip(),
            *candidate.get("aliases", []),
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = clean_text(term)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _resolve_ai_candidate(self, city: City, city_entity_id: str | None, candidate: dict) -> tuple[dict | None, str]:
        search_terms = self._build_ai_search_terms(city, candidate)
        prelim_results: dict[str, dict] = {}
        search_failed = False
        for term in search_terms:
            for result in self.wikidata.search_entities(term, limit=5):
                result_id = result.get("id", "")
                if not result_id:
                    continue
                score = self._score_wikidata_resolution(city, candidate["name"], result, candidate.get("aliases", []))
                current = prelim_results.get(result_id)
                if current is None or score > current["score"]:
                    prelim_results[result_id] = {"result": result, "score": score}

        if not prelim_results:
            return None, "search_no_match"

        best_candidate = None
        best_score = -9999
        best_reason = "low_confidence"
        best_distance_km = None

        ranked_prelim = sorted(prelim_results.values(), key=lambda item: item["score"], reverse=True)[:3]
        entity_ids = [item["result"]["id"] for item in ranked_prelim if item.get("result", {}).get("id")]
        try:
            entity_lookup = self.wikidata.get_entities(entity_ids)
        except RequestException:
            return None, "wikidata_rate_limited"
        for item in ranked_prelim:
            result = item["result"]
            entity = entity_lookup.get(result["id"])
            if not entity:
                search_failed = True
                continue
            score, distance_km, reason = self._score_ai_entity_candidate(city, city_entity_id, candidate, result, entity)
            if score > best_score:
                best_score = score
                best_reason = reason
                best_distance_km = distance_km
                best_candidate = (result, entity)

        if best_candidate is None or best_reason != "resolved":
            if search_failed and best_candidate is None:
                return None, "wikidata_rate_limited"
            return None, best_reason

        best_result, entity = best_candidate
        lat, lng = self._extract_entity_coords(entity)

        sitelinks = entity.get("sitelinks", {})
        wikipedia_title = ""
        preferred_wiki = "eswiki"
        if preferred_wiki in sitelinks:
            wikipedia_title = sitelinks[preferred_wiki].get("title", "")
        elif "enwiki" in sitelinks:
            wikipedia_title = sitelinks["enwiki"].get("title", "")

        description = (
            entity.get("descriptions", {}).get("es", {}).get("value")
            or best_result.get("description", "")
            or ""
        )
        return (
            {
                "source": "wikidata_ai",
                "wikidata_id": best_result["id"],
                "poi_name": clean_text(best_result.get("label", "")) or candidate["name"],
                "lat": lat,
                "lng": lng,
                "description": clean_text(description),
                "type_label": candidate["poi_type_code"],
                "type_code": candidate["poi_type_code"],
                "wikipedia_title": wikipedia_title,
                "sitelinks": len(sitelinks),
                "resolution_score": best_score,
                "distance_km": round(best_distance_km, 2) if best_distance_km is not None else None,
            },
            "resolved",
        )

    def _map_overpass_type_code(self, tags: dict) -> str:
        for key in ("tourism", "historic", "amenity", "building", "place"):
            value = clean_text(str(tags.get(key, ""))).lower()
            if not value:
                continue
            mapped = self.overpass_tag_type_map.get(f"{key}:{value}")
            if mapped:
                return mapped
        return self._map_poi_type_code(
            clean_text(str(tags.get("name", ""))),
            "",
            " ".join(
                [
                    clean_text(str(tags.get("tourism", ""))),
                    clean_text(str(tags.get("historic", ""))),
                    clean_text(str(tags.get("amenity", ""))),
                    clean_text(str(tags.get("building", ""))),
                ]
            ),
        )

    def _score_tourism_candidate(
        self,
        *,
        name: str,
        description: str,
        type_code: str,
        type_label: str,
        sitelinks: int,
        wikipedia_title: str,
    ) -> int:
        text = " ".join([clean_text(name).lower(), clean_text(description).lower(), clean_text(type_label).lower()]).strip()
        score = self.tourism_type_scores.get(type_code, 0)
        score += min(max(sitelinks, 0), 120)

        if wikipedia_title:
            score += 15
        if len(clean_text(description)) > 18:
            score += 10

        for token, value in self.tourism_positive_terms.items():
            if token in text:
                score += value
        for token, value in self.tourism_negative_terms.items():
            if token in text:
                score += value

        if "lista de" in text or "list of" in text:
            score -= 120
        if "edificio" in text and type_code == "building":
            score -= 20
        return score

    def _is_map_candidate(self, score: int, type_code: str) -> bool:
        threshold = 35 if type_code != "building" else 45
        return score >= threshold

    def _is_featured_candidate(self, score: int, type_code: str) -> bool:
        threshold = 55 if type_code != "building" else 75
        return score >= threshold

    def _build_city_entity_import_query(self, city_entity_id: str, limit: int) -> str:
        safe_limit = max(1, min(int(limit), 80))
        fetch_limit = min(max(safe_limit * 3, 80), 240)
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>

SELECT ?poi ?poiLabel ?poiDescription ?coord ?poiTypeLabel ?resolvedArticle ?sitelinks WHERE {{
  {{
    ?poi wdt:P131/wdt:P131* wd:{city_entity_id} .
  }}
  UNION
  {{
    ?poi wdt:P276/wdt:P131* wd:{city_entity_id} .
  }}
  ?poi wdt:P625 ?coord .
  ?poi wdt:P31/wdt:P279* ?poiType .
  VALUES ?poiType {{
    wd:Q570116
    wd:Q33506
    wd:Q207694
    wd:Q16970
    wd:Q2977
    wd:Q41176
    wd:Q174782
    wd:Q4989906
    wd:Q16560
    wd:Q839954
    wd:Q12280
    wd:Q811979
    wd:Q24354
  }}
  OPTIONAL {{
    ?article schema:about ?poi ;
             schema:isPartOf <https://es.wikipedia.org/> .
  }}
  OPTIONAL {{
    ?articleEn schema:about ?poi ;
               schema:isPartOf <https://en.wikipedia.org/> .
  }}
  BIND(COALESCE(?article, ?articleEn) AS ?resolvedArticle)
  OPTIONAL {{ ?poi wikibase:sitelinks ?sitelinks . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {fetch_limit}
"""

    def _build_radius_import_query(self, city: City, radius_km: float, limit: int) -> str:
        safe_limit = max(1, min(int(limit), 80))
        fetch_limit = min(max(safe_limit * 2, 80), 200)
        safe_radius = min(radius_km, 12.0)
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX schema: <http://schema.org/>

SELECT ?poi ?poiLabel ?poiDescription ?coord ?poiTypeLabel ?resolvedArticle ?sitelinks WHERE {{
  SERVICE wikibase:around {{
    ?poi wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({city.lng} {city.lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{safe_radius}" .
  }}
  ?poi wdt:P31/wdt:P279* ?poiType .
  VALUES ?poiType {{
    wd:Q570116
    wd:Q33506
    wd:Q207694
    wd:Q16970
    wd:Q2977
    wd:Q41176
    wd:Q174782
    wd:Q4989906
    wd:Q16560
    wd:Q839954
    wd:Q12280
    wd:Q811979
    wd:Q24354
  }}
  OPTIONAL {{
    ?article schema:about ?poi ;
             schema:isPartOf <https://es.wikipedia.org/> .
  }}
  OPTIONAL {{
    ?articleEn schema:about ?poi ;
               schema:isPartOf <https://en.wikipedia.org/> .
  }}
  BIND(COALESCE(?article, ?articleEn) AS ?resolvedArticle)
  OPTIONAL {{ ?poi wikibase:sitelinks ?sitelinks . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {fetch_limit}
"""

    def _build_overpass_map_query(self, city: City, radius_km: float, limit: int) -> str:
        safe_limit = max(20, min(int(limit) * 4, 180))
        safe_radius_m = int(min(max(radius_km, 1.5), 8.0) * 1000)
        return f"""
[out:json][timeout:20];
(
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["tourism"~"^(attraction|museum|gallery|artwork)$"]["name"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["historic"~"^(monument|memorial|castle|archaeological_site|ruins)$"]["name"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["historic"~"^(yes|roman_road|citywalls|fort|aqueduct)$"]["name"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["amenity"="place_of_worship"]["name"]["wikidata"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["amenity"="place_of_worship"]["name"]["heritage"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["building"~"^(cathedral|church|synagogue|chapel|monastery)$"]["name"]["wikidata"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["building"~"^(cathedral|church|synagogue|chapel|monastery)$"]["name"]["wikipedia"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["building"~"^(palace|public|yes|historic)$"]["name"]["wikidata"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["building"~"^(palace|public|yes|historic)$"]["name"]["heritage"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["place"="square"]["name"]["wikidata"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["place"="square"]["name"]["wikipedia"];
  nwr(around:{safe_radius_m},{float(city.lat)},{float(city.lng)})["heritage"]["name"];
);
out center {safe_limit};
"""

    def _normalize_overpass_element(self, element: dict) -> dict | None:
        tags = element.get("tags", {}) or {}
        name = clean_text(str(tags.get("name", "")))
        if not name:
            return None
        lat = element.get("lat")
        lng = element.get("lon")
        center = element.get("center", {}) or {}
        if lat is None:
            lat = center.get("lat")
        if lng is None:
            lng = center.get("lon")
        if lat is None or lng is None:
            return None
        wikipedia_title = clean_text(str(tags.get("wikipedia", "")))
        if ":" in wikipedia_title:
            wikipedia_title = wikipedia_title.split(":", 1)[1]
        description = clean_text(
            str(tags.get("description") or tags.get("historic") or tags.get("tourism") or tags.get("amenity") or "")
        )
        type_code = self._map_overpass_type_code(tags)
        wikidata_id = clean_text(str(tags.get("wikidata", "")))
        return {
            "source_id": f"osm:{element.get('type', '')}:{element.get('id', '')}",
            "name": name,
            "lat": lat,
            "lng": lng,
            "description": description,
            "type_code": type_code,
            "type_label": clean_text(
                " ".join(
                    [
                        str(tags.get("tourism", "")),
                        str(tags.get("historic", "")),
                        str(tags.get("amenity", "")),
                        str(tags.get("building", "")),
                        str(tags.get("place", "")),
                    ]
                )
            ),
            "wikipedia_title": wikipedia_title,
            "wikidata_id": wikidata_id,
        }

    def _build_overpass_name_query(self, city: City, name: str, aliases: list[str]) -> str:
        radius_m = 12000
        terms = [clean_text(name), *[clean_text(alias) for alias in aliases[:3]]]
        clauses = []
        for term in terms:
            if not term:
                continue
            escaped = term.replace('"', '\\"')
            clauses.append(
                f'nwr(around:{radius_m},{float(city.lat)},{float(city.lng)})["name"="{escaped}"];'
            )
        if not clauses:
            return ""
        body = "\n  ".join(clauses)
        return f"""
[out:json][timeout:20];
(
  {body}
);
out center 10;
"""

    def _resolve_candidate_with_overpass(self, city: City, candidate: dict) -> dict | None:
        query = self._build_overpass_name_query(city, candidate["name"], candidate.get("aliases", []))
        if not query:
            return None
        try:
            elements = self.overpass.query(query)
        except RequestException:
            return None
        normalized = [item for item in (self._normalize_overpass_element(element) for element in elements) if item]
        if not normalized:
            return None
        normalized.sort(
            key=lambda item: (
                0 if clean_text(item["name"]).lower() == clean_text(candidate["name"]).lower() else 1,
                self._distance_km(city.lat, city.lng, item["lat"], item["lng"]) or 9999,
            )
        )
        best = normalized[0]
        return {
            "source": "overpass",
            "source_id": best["source_id"],
            "wikidata_id": best["wikidata_id"],
            "poi_name": best["name"],
            "lat": best["lat"],
            "lng": best["lng"],
            "description": best["description"],
            "type_label": best["type_label"],
            "type_code": candidate["poi_type_code"] or best["type_code"],
            "wikipedia_title": best["wikipedia_title"],
            "sitelinks": 0,
            "resolution_score": None,
            "distance_km": round(self._distance_km(city.lat, city.lng, best["lat"], best["lng"]) or 0, 2),
        }

    def enrich_city_pending_pois(self, city_id: int, limit: int = 150) -> None:
        with session_scope() as db:
            city = db.get(City, city_id)
            if city is None:
                return
            city_entity_id = self._resolve_city_entity_id(city)
            pending_pois = list(
                db.scalars(
                    select(Poi)
                    .where(Poi.city_id == city.id)
                    .order_by(Poi.id.asc())
                ).all()
            )
            processed = 0
            for poi in pending_pois:
                metadata = dict(poi.metadata_json or {})
                status = metadata.get("import_status", "")
                if status not in {"pending_wikidata", "retry_wikidata", "rate_limited_retry"}:
                    continue
                candidate = self._build_pending_candidate(poi)
                metadata["resolution_attempts"] = int(metadata.get("resolution_attempts", 0)) + 1
                resolved, reason = self._resolve_ai_candidate(city, city_entity_id, candidate)
                if resolved is not None:
                    poi.lat = Decimal(str(resolved["lat"])) if resolved["lat"] is not None else poi.lat
                    poi.lng = Decimal(str(resolved["lng"])) if resolved["lng"] is not None else poi.lng
                    poi.wikidata_id = resolved["wikidata_id"] or poi.wikidata_id
                    poi.wikipedia_title = resolved["wikipedia_title"] or poi.wikipedia_title
                    if resolved["description"]:
                        poi.short_description = resolved["description"]
                    poi.source_of_truth = "wikidata"
                    metadata["import_status"] = "resolved"
                    metadata["import_tier"] = "featured"
                    metadata["catalog_source"] = resolved["source"]
                    metadata["resolution_reason"] = "wikidata"
                    metadata["resolution_score"] = resolved.get("resolution_score")
                    metadata["distance_km"] = resolved.get("distance_km")
                else:
                    overpass_resolved = None
                    if reason not in {"wikidata_rate_limited"}:
                        overpass_resolved = self._resolve_candidate_with_overpass(city, candidate)
                    if overpass_resolved is not None:
                        poi.lat = Decimal(str(overpass_resolved["lat"])) if overpass_resolved["lat"] is not None else poi.lat
                        poi.lng = Decimal(str(overpass_resolved["lng"])) if overpass_resolved["lng"] is not None else poi.lng
                        poi.wikipedia_title = overpass_resolved["wikipedia_title"] or poi.wikipedia_title
                        poi.source_of_truth = "overpass"
                        metadata["import_status"] = "resolved"
                        metadata["import_tier"] = "map"
                        metadata["catalog_source"] = "overpass"
                        metadata["source_id"] = overpass_resolved.get("source_id", "")
                        metadata["resolution_reason"] = "overpass_fallback"
                        metadata["distance_km"] = overpass_resolved.get("distance_km")
                    elif reason == "wikidata_rate_limited":
                        metadata["import_status"] = "rate_limited_retry"
                        metadata["resolution_reason"] = reason
                        poi.metadata_json = metadata
                        db.flush()
                        self.logger.warning("Wikidata rate limited while enriching city %s; stopping batch", city.id)
                        break
                    else:
                        metadata["import_status"] = "unresolved"
                        metadata["resolution_reason"] = reason
                poi.metadata_json = metadata
                processed += 1
                db.flush()
                if processed >= limit:
                    break
                time.sleep(0.15)

    def start_pending_enrichment(self, city_id: int, limit: int = 150) -> bool:
        with self._enrichment_lock:
            if city_id in self._active_enrichment_jobs:
                return False
            self._active_enrichment_jobs.add(city_id)

        def runner() -> None:
            try:
                self.enrich_city_pending_pois(city_id, limit=limit)
            except Exception as exc:
                self.logger.exception("Background enrichment failed for city %s: %s", city_id, exc)
            finally:
                with self._enrichment_lock:
                    self._active_enrichment_jobs.discard(city_id)

        thread = threading.Thread(
            target=runner,
            name=f"catalog-enrich-{city_id}",
            daemon=True,
        )
        thread.start()
        return True

    def import_city_pois(self, city_id: int, radius_km: float = 8.0, limit: int = 40, use_ai_candidates: bool = True) -> tuple[int, int, int, dict, list[PoiResponse], str]:
        with session_scope() as db:
            city = db.get(City, city_id)
            if city is None:
                raise CatalogError("Ciudad no encontrada")
            if city.lat is None or city.lng is None:
                raise CatalogError("La ciudad no tiene coordenadas guardadas")
            type_lookup = {item.code: item for item in db.scalars(select(PoiType)).all()}

            bindings: list[dict] = []
            overpass_candidates: list[dict] = []
            ai_candidates: list[dict] = []
            city_entity_id = self._resolve_city_entity_id(city)
            if use_ai_candidates and self.openai.is_configured():
                try:
                    ai_candidates = self._generate_ai_candidates(city, limit=min(limit, 150))
                except CatalogError:
                    ai_candidates = []

            if ai_candidates:
                imported_count, updated_count, skipped_count, imported_rows, stats = self._upsert_ai_seed_candidates(
                    db,
                    city,
                    type_lookup,
                    ai_candidates,
                    min(limit, 150),
                )
                return imported_count, updated_count, skipped_count, stats, imported_rows, city.name

            if city_entity_id and not ai_candidates:
                try:
                    bindings.extend(self.wikidata.run_sparql(self._build_city_entity_import_query(city_entity_id, limit=limit)))
                except RequestException:
                    pass

            if not ai_candidates:
                try:
                    bindings.extend(self.wikidata.run_sparql(self._build_radius_import_query(city, radius_km=radius_km, limit=limit)))
                except RequestException as exc:
                    if not bindings:
                        raise CatalogError(
                            "Wikidata ha tardado demasiado al importar POIs. Prueba otra vez o reduce el radio de búsqueda."
                        ) from exc
                try:
                    elements = self.overpass.query(self._build_overpass_map_query(city, radius_km=radius_km, limit=limit))
                    overpass_candidates = [item for item in (self._normalize_overpass_element(element) for element in elements) if item]
                except RequestException:
                    overpass_candidates = []
            imported_count = 0
            updated_count = 0
            skipped_count = 0
            imported_rows: list[PoiResponse] = []
            seen_ids: set[str] = set()
            ranked_candidates: list[tuple[int, dict]] = []
            stats = {
                "mode": "ai_candidates" if ai_candidates else "catalog_sources",
                "ai": {
                    "proposed_count": len(ai_candidates),
                    "resolved_count": 0,
                    "rejected_count": 0,
                    "duplicate_count": 0,
                    "reasons": {},
                },
                "sources": {
                    "wikidata_rows": len(bindings),
                    "overpass_rows": len(overpass_candidates),
                },
                "ranked_candidate_count": 0,
                "selected_candidate_count": 0,
            }

            def bump_reason(bucket: dict, reason: str) -> None:
                bucket[reason] = int(bucket.get(reason, 0)) + 1

            for candidate in ai_candidates:
                resolved, reason = self._resolve_ai_candidate(city, city_entity_id, candidate)
                if resolved is None:
                    stats["ai"]["rejected_count"] += 1
                    bump_reason(stats["ai"]["reasons"], reason)
                    continue
                dedupe_key = resolved["wikidata_id"] or slugify(resolved["poi_name"])
                if dedupe_key in seen_ids:
                    stats["ai"]["duplicate_count"] += 1
                    bump_reason(stats["ai"]["reasons"], "duplicate_resolved")
                    continue
                seen_ids.add(dedupe_key)
                score = self._score_tourism_candidate(
                    name=resolved["poi_name"],
                    description=resolved["description"],
                    type_code=resolved["type_code"],
                    type_label=resolved["type_label"],
                    sitelinks=resolved["sitelinks"],
                    wikipedia_title=resolved["wikipedia_title"],
                )
                stats["ai"]["resolved_count"] += 1
                ranked_candidates.append((score, resolved))

            for item in bindings:
                poi_uri = item.get("poi", {}).get("value", "")
                wikidata_id = poi_uri.rsplit("/", 1)[-1] if poi_uri else ""
                if not wikidata_id or wikidata_id in seen_ids:
                    continue
                seen_ids.add(wikidata_id)

                poi_name = clean_text(item.get("poiLabel", {}).get("value", ""))
                if not poi_name:
                    continue

                lat, lng = self._wkt_point_to_coords(item.get("coord", {}).get("value", ""))
                description = clean_text(item.get("poiDescription", {}).get("value", ""))
                type_label = clean_text(item.get("poiTypeLabel", {}).get("value", ""))
                type_code = self._map_poi_type_code(poi_name, type_label, description)
                wikipedia_title = item.get("resolvedArticle", {}).get("value", "").rsplit("/", 1)[-1]
                try:
                    sitelinks = int(float(item.get("sitelinks", {}).get("value", "0") or "0"))
                except ValueError:
                    sitelinks = 0
                score = self._score_tourism_candidate(
                    name=poi_name,
                    description=description,
                    type_code=type_code,
                    type_label=type_label,
                    sitelinks=sitelinks,
                    wikipedia_title=wikipedia_title,
                )
                if not self._is_map_candidate(score, type_code):
                    continue
                ranked_candidates.append(
                    (
                        score,
                        {
                            "source": "wikidata",
                            "wikidata_id": wikidata_id,
                            "poi_name": poi_name,
                            "lat": lat,
                            "lng": lng,
                            "description": description,
                            "type_label": type_label,
                            "type_code": type_code,
                            "wikipedia_title": wikipedia_title,
                            "sitelinks": sitelinks,
                        },
                    )
                )

            for candidate in overpass_candidates:
                wikidata_id = candidate["wikidata_id"]
                dedupe_key = wikidata_id or slugify(candidate["name"])
                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)
                score = self._score_tourism_candidate(
                    name=candidate["name"],
                    description=candidate["description"],
                    type_code=candidate["type_code"],
                    type_label=candidate["type_label"],
                    sitelinks=0,
                    wikipedia_title=candidate["wikipedia_title"],
                )
                if not self._is_map_candidate(score, candidate["type_code"]):
                    continue
                ranked_candidates.append(
                    (
                        score,
                        {
                            "source": "osm",
                            "source_id": candidate["source_id"],
                            "wikidata_id": wikidata_id,
                            "poi_name": candidate["name"],
                            "lat": candidate["lat"],
                            "lng": candidate["lng"],
                            "description": candidate["description"],
                            "type_label": candidate["type_label"],
                            "type_code": candidate["type_code"],
                            "wikipedia_title": candidate["wikipedia_title"],
                            "sitelinks": 0,
                        },
                    )
                )

            ranked_candidates.sort(key=lambda item: (-item[0], item[1]["poi_name"].lower()))
            featured_limit = min(max(limit // 2, 4), 8)
            selected_candidates = ranked_candidates[:limit]
            stats["ranked_candidate_count"] = len(ranked_candidates)
            stats["selected_candidate_count"] = len(selected_candidates)

            for index, (score, candidate) in enumerate(selected_candidates):
                wikidata_id = candidate["wikidata_id"]
                poi_name = candidate["poi_name"]
                lat = candidate["lat"]
                lng = candidate["lng"]
                description = candidate["description"]
                type_label = candidate["type_label"]
                type_code = candidate["type_code"]
                wikipedia_title = candidate["wikipedia_title"]
                poi_type = type_lookup.get(type_code)
                is_featured = (
                    candidate.get("source") in {"wikidata", "wikidata_ai"}
                    and index < featured_limit
                    and self._is_featured_candidate(score, type_code)
                )

                existing = db.scalar(
                    select(Poi)
                    .where(Poi.city_id == city.id)
                    .where(or_(Poi.wikidata_id == wikidata_id, Poi.slug == slugify(poi_name)))
                )
                if existing is None:
                    existing = Poi(
                        city_id=city.id,
                        poi_type_id=poi_type.id if poi_type else None,
                        slug=slugify(poi_name),
                        name=poi_name,
                        lat=Decimal(str(lat)) if lat is not None else None,
                        lng=Decimal(str(lng)) if lng is not None else None,
                        short_description=description,
                        long_description="",
                        source_of_truth="wikidata",
                        wikidata_id=wikidata_id,
                        wikipedia_title=wikipedia_title,
                        is_active=True,
                        metadata_json={
                            "imported_from": (
                                "wikidata_ai"
                                if candidate.get("source") == "wikidata_ai"
                                else "wikidata_sparql"
                                if candidate.get("source") == "wikidata"
                                else "overpass"
                            ),
                            "type_label": type_label,
                            "tourism_score": score,
                            "wikipedia_title": wikipedia_title,
                            "featured": is_featured,
                            "import_tier": "featured" if is_featured else "map",
                            "catalog_source": candidate.get("source"),
                            "source_id": candidate.get("source_id", ""),
                            "resolution_score": candidate.get("resolution_score"),
                            "distance_km": candidate.get("distance_km"),
                        },
                    )
                    db.add(existing)
                    db.flush()
                    imported_count += 1
                else:
                    existing.poi_type_id = poi_type.id if poi_type else existing.poi_type_id
                    existing.lat = Decimal(str(lat)) if lat is not None else existing.lat
                    existing.lng = Decimal(str(lng)) if lng is not None else existing.lng
                    existing.short_description = description or existing.short_description
                    existing.source_of_truth = (
                        "wikidata"
                        if candidate.get("source") in {"wikidata", "wikidata_ai"}
                        else existing.source_of_truth
                    )
                    existing.wikidata_id = wikidata_id or existing.wikidata_id
                    existing.wikipedia_title = wikipedia_title or existing.wikipedia_title
                    current_meta = dict(existing.metadata_json or {})
                    current_meta["imported_from"] = (
                        "wikidata_ai"
                        if candidate.get("source") == "wikidata_ai"
                        else "wikidata_sparql"
                        if candidate.get("source") == "wikidata"
                        else "overpass"
                    )
                    current_meta["type_label"] = type_label
                    current_meta["tourism_score"] = score
                    current_meta["wikipedia_title"] = wikipedia_title
                    current_meta["featured"] = is_featured
                    current_meta["import_tier"] = "featured" if is_featured else "map"
                    current_meta["catalog_source"] = candidate.get("source")
                    current_meta["source_id"] = candidate.get("source_id", current_meta.get("source_id", ""))
                    current_meta["resolution_score"] = candidate.get("resolution_score", current_meta.get("resolution_score"))
                    current_meta["distance_km"] = candidate.get("distance_km", current_meta.get("distance_km"))
                    existing.metadata_json = current_meta
                    updated_count += 1

                imported_rows.append(self._poi_to_schema(existing, poi_type))

            skipped_count = max(0, stats["ai"]["rejected_count"] + stats["ai"]["duplicate_count"])
            return imported_count, updated_count, skipped_count, stats, imported_rows, city.name


catalog_service = CatalogService()
