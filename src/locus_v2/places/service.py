"""Nearby-place search: catalog rows first, live Google Places to fill gaps.

Ported from V1 `app/services/poi_service.py`, with two deliberate changes:

- Results are returned as `SessionPoi`, the shape the map session and the
  Ionic app already speak, instead of V1's separate `POI` schema.
- Catalog names/descriptions come back **localized** (`localized_field`).
  V1 always answered with the Spanish `name` column because its catalog had
  no per-language columns; V2's does, so a French user gets French labels
  on the same rows.
"""

from decimal import Decimal
from math import cos, radians, sqrt

from sqlalchemy import Select, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.catalog.models import Poi
from locus_v2.config import Settings
from locus_v2.places.client import GooglePlacesClient
from locus_v2.sessions.models import SessionPoi
from locus_v2.shared.prompting import localized_field
from locus_v2.shared.text import clean_text, slugify

# A degree of latitude is ~111.32 km everywhere; a degree of longitude shrinks
# with the cosine of the latitude. The 0.2 floor keeps the maths finite near
# the poles instead of dividing by ~0. Verbatim from V1.
_KM_PER_DEGREE = 111.32
_MIN_LNG_FACTOR = 0.2

_GENERIC_QUERIES = {
    "",
    "lugares turisticos",
    "lugares turísticos",
    "que ver",
    "qué ver",
    "recomiendame algo",
    "recomiéndame algo",
}

_SERVICE_LABELS = (
    "restaurante", "restaurant", "bar", "pub", "cafe", "café", "hotel",
    "hostel", "farmacia", "pharmacy", "supermercado", "taxi", "garden",
)


def distance_km(lat_a: float, lng_a: float, lat_b: float, lng_b: float) -> float:
    lat_factor = _KM_PER_DEGREE
    lng_factor = _KM_PER_DEGREE * max(cos(radians(lat_a)), _MIN_LNG_FACTOR)
    return sqrt(((lat_b - lat_a) * lat_factor) ** 2 + ((lng_b - lng_a) * lng_factor) ** 2)


def is_generic_query(query: str) -> bool:
    return clean_text(query).lower() in _GENERIC_QUERIES


def looks_like_service(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(token in lowered for token in _SERVICE_LABELS)


class PlaceSearchService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.places = GooglePlacesClient(settings)

    async def search_catalog(
        self,
        *,
        query: str,
        lat: float | None,
        lng: float | None,
        locale: str,
        limit: int = 5,
        radius_km: float = 10.0,
    ) -> list[SessionPoi]:
        statement: Select[tuple[Poi]] = select(Poi).where(Poi.is_active.is_(True))
        if lat is not None and lng is not None:
            lat_delta = radius_km / _KM_PER_DEGREE
            lng_delta = radius_km / max(
                _KM_PER_DEGREE * max(cos(radians(lat)), _MIN_LNG_FACTOR), 1
            )
            statement = statement.where(
                Poi.lat.is_not(None),
                Poi.lng.is_not(None),
                Poi.lat.between(Decimal(str(lat - lat_delta)), Decimal(str(lat + lat_delta))),
                Poi.lng.between(Decimal(str(lng - lng_delta)), Decimal(str(lng + lng_delta))),
            )
        if not is_generic_query(query):
            token = f"%{clean_text(query)}%"
            statement = statement.where(
                or_(
                    Poi.name.ilike(token),
                    Poi.slug.ilike(token),
                    Poi.short_description.ilike(token),
                    Poi.long_description.ilike(token),
                )
            )

        # Over-fetch, then rank by real distance in Python: the bounding box
        # above is a square, not a circle, and MySQL has no cheap geo sort here.
        rows = list(
            (await self.session.scalars(statement.limit(max(limit * 6, 20)))).unique().all()
        )
        if lat is not None and lng is not None:
            rows.sort(
                key=lambda row: (
                    distance_km(lat, lng, float(row.lat), float(row.lng))
                    if row.lat is not None and row.lng is not None
                    else 9999.0
                )
            )
        else:
            rows.sort(key=lambda row: row.name.lower())
        return [_catalog_poi(row, locale) for row in rows[:limit]]

    async def search_landmarks(
        self,
        *,
        query: str,
        lat: float | None,
        lng: float | None,
        locale: str,
        limit: int = 5,
    ) -> list[SessionPoi]:
        """Tourism/landmark search: catalog first, Google Places to fill gaps."""
        results = await self.search_catalog(
            query=query, lat=lat, lng=lng, locale=locale, limit=limit
        )
        if not is_generic_query(query) and len(results) < limit:
            external = await self.places.search_places(
                query=query, lat=lat, lng=lng, limit=limit, locale=locale
            )
            results = _merge_unique(results, [SessionPoi(**item) for item in external], limit)

        # Landmarks only: drop hospitality/services and anything implausibly far.
        landmarks: list[SessionPoi] = []
        for poi in results:
            haystack = f"{poi.name} {poi.description} {poi.summary}"
            if looks_like_service(haystack):
                continue
            if lat is not None and lng is not None and distance_km(lat, lng, poi.lat, poi.lng) > 8:
                continue
            landmarks.append(
                poi.model_copy(
                    update={
                        "context_kind": "catalog" if not poi.is_ephemeral else "tourism_candidate"
                    }
                )
            )
        return landmarks[:limit]

    async def search_services(
        self,
        *,
        query: str,
        lat: float | None,
        lng: float | None,
        locale: str = "",
        limit: int = 5,
    ) -> list[SessionPoi]:
        """Restaurants, bars, pharmacies... — live Google Places only.

        These are deliberately never persisted: the catalog is for places
        worth a guided visit, not for where to have lunch.
        """
        external = await self.places.search_places(
            query=query, lat=lat, lng=lng, limit=limit, locale=locale
        )
        context_kind = _service_context_kind(query)
        return [
            SessionPoi(**item).model_copy(update={"context_kind": context_kind})
            for item in external
        ]


def _service_context_kind(need: str) -> str:
    lowered = clean_text(need).lower()
    hospitality = (
        "restaurante", "comer", "cenar", "bar", "cerveza", "cafe", "café",
        "desayuno", "pizza", "pasta",
    )
    return "hospitality" if any(token in lowered for token in hospitality) else "service"


def _catalog_poi(row: Poi, locale: str) -> SessionPoi:
    language = locale.split("-", 1)[0].lower()
    name = localized_field(row.names_json, locale, language) or row.name
    short = (
        localized_field(row.short_descriptions_json, locale, language)
        or row.short_description
    )
    return SessionPoi(
        id=str(row.id),
        name=name,
        lat=float(row.lat) if row.lat is not None else 0.0,
        lng=float(row.lng) if row.lng is not None else 0.0,
        description=short or row.long_description or "",
        summary=row.long_description or short or "",
        source_of_truth="catalog",
        is_ephemeral=False,
        google_place_id=row.google_place_id or "",
        context_kind="catalog",
    )


def _merge_unique(
    primary: list[SessionPoi], secondary: list[SessionPoi], limit: int
) -> list[SessionPoi]:
    merged: list[SessionPoi] = []
    seen: set[str] = set()
    for poi in [*primary, *secondary]:
        key = slugify(poi.name)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(poi)
        if len(merged) >= limit:
            break
    return merged
