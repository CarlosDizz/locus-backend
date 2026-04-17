from __future__ import annotations

import re
from decimal import Decimal

from sqlalchemy import Select, or_, select

from app.clients.wikidata_client import WikidataClient
from app.db.models import City, Poi, PoiType
from app.db.session import session_scope
from app.schemas.catalog import CityCreateRequest, CityResponse, PoiCreateRequest, PoiResponse, PoiTypeResponse, PoiUpdateRequest
from app.utils.text import clean_text, slugify


class CatalogError(RuntimeError):
    pass


class CatalogService:
    def __init__(self) -> None:
        self.wikidata = WikidataClient()
        self.poi_type_map = {
            "tourist attraction": "monument",
            "museum": "museum",
            "art museum": "museum",
            "church building": "church",
            "cathedral": "church",
            "basilica": "church",
            "square": "square",
            "monument": "monument",
            "palace": "building",
            "archaeological site": "archaeological_site",
            "bridge": "building",
            "building": "building",
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
        search = self.wikidata.search_entity(f"{query} city", limit=1) or self.wikidata.search_entity(query, limit=1)
        if search is None:
            raise CatalogError("No he podido resolver la ciudad en Wikidata")
        entity = self.wikidata.get_entity(search["id"])
        coords_claim = (entity.get("claims", {}).get("P625") or [])
        lat = None
        lng = None
        if coords_claim:
            value = coords_claim[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
            lat = value.get("latitude")
            lng = value.get("longitude")
        city_name = (
            entity.get("labels", {}).get("es", {}).get("value")
            or search.get("label")
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

    def _map_poi_type_code(self, type_label: str, fallback_description: str) -> str:
        label = (type_label or "").strip().lower()
        description = (fallback_description or "").strip().lower()
        for token, code in self.poi_type_map.items():
            if token in label or token in description:
                return code
        return "building"

    def import_city_pois(self, city_id: int, radius_km: float = 8.0, limit: int = 80) -> tuple[int, int, int, list[PoiResponse], str]:
        with session_scope() as db:
            city = db.get(City, city_id)
            if city is None:
                raise CatalogError("Ciudad no encontrada")
            if city.lat is None or city.lng is None:
                raise CatalogError("La ciudad no tiene coordenadas guardadas")
            type_lookup = {item.code: item for item in db.scalars(select(PoiType)).all()}

            query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX schema: <http://schema.org/>

SELECT ?poi ?poiLabel ?poiDescription ?coord ?typeLabel ?article WHERE {{
  SERVICE wikibase:around {{
    ?poi wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({city.lng} {city.lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius_km}" .
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
    wd:Q41176
    wd:Q12280
    wd:Q811979
  }}
  OPTIONAL {{
    ?article schema:about ?poi ;
             schema:isPartOf <https://es.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
}}
LIMIT {int(limit)}
"""
            bindings = self.wikidata.run_sparql(query)
            imported_count = 0
            updated_count = 0
            skipped_count = 0
            imported_rows: list[PoiResponse] = []
            seen_ids: set[str] = set()

            for item in bindings:
                poi_uri = item.get("poi", {}).get("value", "")
                wikidata_id = poi_uri.rsplit("/", 1)[-1] if poi_uri else ""
                if not wikidata_id or wikidata_id in seen_ids:
                    skipped_count += 1
                    continue
                seen_ids.add(wikidata_id)

                poi_name = clean_text(item.get("poiLabel", {}).get("value", ""))
                if not poi_name:
                    skipped_count += 1
                    continue

                lat, lng = self._wkt_point_to_coords(item.get("coord", {}).get("value", ""))
                description = clean_text(item.get("poiDescription", {}).get("value", ""))
                type_label = clean_text(item.get("typeLabel", {}).get("value", ""))
                type_code = self._map_poi_type_code(type_label, description)
                poi_type = type_lookup.get(type_code)
                wikipedia_title = item.get("article", {}).get("value", "").rsplit("/", 1)[-1]

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
                        metadata_json={"imported_from": "wikidata_sparql", "type_label": type_label},
                    )
                    db.add(existing)
                    db.flush()
                    imported_count += 1
                else:
                    existing.poi_type_id = poi_type.id if poi_type else existing.poi_type_id
                    existing.lat = Decimal(str(lat)) if lat is not None else existing.lat
                    existing.lng = Decimal(str(lng)) if lng is not None else existing.lng
                    existing.short_description = description or existing.short_description
                    existing.source_of_truth = "wikidata"
                    existing.wikidata_id = wikidata_id or existing.wikidata_id
                    existing.wikipedia_title = wikipedia_title or existing.wikipedia_title
                    current_meta = dict(existing.metadata_json or {})
                    current_meta["imported_from"] = "wikidata_sparql"
                    current_meta["type_label"] = type_label
                    existing.metadata_json = current_meta
                    updated_count += 1

                imported_rows.append(self._poi_to_schema(existing, poi_type))

            return imported_count, updated_count, skipped_count, imported_rows, city.name


catalog_service = CatalogService()
