from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel


class AdminCitySummary(BaseModel):
    id: int
    public_id: str
    name: str
    names: dict[str, str]
    slug: str
    country_code: str
    lat: Decimal | None
    lng: Decimal | None
    source: str
    poi_count: int
    active_poi_count: int


class AdminCityList(BaseModel):
    total: int
    items: list[AdminCitySummary]


class AdminPoiSummary(BaseModel):
    id: int
    public_id: str
    name: str
    slug: str
    type_code: str | None
    type_name: str | None
    lat: Decimal | None
    lng: Decimal | None
    source_of_truth: str
    is_active: bool


class AdminPoiPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AdminPoiSummary]


class AdminPoiDetail(AdminPoiSummary):
    city_id: int | None
    city_name: str | None
    names: dict[str, str]
    short_description: str
    short_descriptions: dict[str, str]
    long_description: str
    wikidata_id: str
    wikipedia_title: str
    google_place_id: str
    metadata: dict[str, Any]


class AdminCatalogReader(Protocol):
    async def cities(self, *, query: str, limit: int) -> AdminCityList: ...

    async def pois(
        self,
        *,
        city_id: int | None,
        query: str,
        active: bool | None,
        limit: int,
        offset: int,
    ) -> AdminPoiPage: ...

    async def poi_detail(self, poi_id: int) -> AdminPoiDetail | None: ...


class AdminCatalogQueryService:
    def __init__(self, reader: AdminCatalogReader) -> None:
        self._reader = reader

    async def list_cities(self, *, query: str = "", limit: int = 100) -> AdminCityList:
        return await self._reader.cities(query=query.strip(), limit=min(max(limit, 1), 300))

    async def list_pois(
        self,
        *,
        city_id: int | None,
        query: str = "",
        active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AdminPoiPage:
        return await self._reader.pois(
            city_id=city_id,
            query=query.strip(),
            active=active,
            limit=min(max(limit, 1), 250),
            offset=max(offset, 0),
        )

    async def detail(self, poi_id: int) -> AdminPoiDetail | None:
        return await self._reader.poi_detail(poi_id)
