"""Wire models matching app/schemas/catalog.py, independent of the V1 runtime."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CityResponse(BaseModel):
    id: int
    slug: str
    name: str
    display_name: str
    names: dict[str, Any] = Field(default_factory=dict)
    country_code: str
    lat: float | None = None
    lng: float | None = None
    source: str
    created_at: datetime


class PoiTypeResponse(BaseModel):
    id: int
    code: str
    name: str
    description: str


class PoiResponse(BaseModel):
    id: int
    city_id: int | None = None
    poi_type_id: int | None = None
    poi_type_code: str | None = None
    poi_type_name: str | None = None
    slug: str
    name: str
    display_name: str
    names: dict[str, Any] = Field(default_factory=dict)
    lat: float | None = None
    lng: float | None = None
    short_description: str
    short_descriptions: dict[str, Any] = Field(default_factory=dict)
    long_description: str
    source_of_truth: str
    wikidata_id: str
    wikipedia_title: str
    google_place_id: str
    is_active: bool
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class PoiDocumentationResponse(BaseModel):
    poi: PoiResponse | None = None
    documentation: dict[str, Any]
    resolved_from_catalog: bool = False


class PoiAccessLinkResponse(BaseModel):
    title: str
    description: str
    url: str
    kind: str
    query: str
    provider: str
    tracking_status: str


class PoiAccessLinksResponse(BaseModel):
    poi_id: int
    poi_name: str
    eligible: bool
    reason: str
    links: list[PoiAccessLinkResponse] = Field(default_factory=list)


class CityBootstrapFromLocationRequest(BaseModel):
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lng: float = Field(ge=-180, le=180, allow_inf_nan=False)
    radius_km: float = Field(default=8.0, gt=0, le=50, allow_inf_nan=False)
    limit: int = Field(default=80, gt=0, le=150)
    use_ai_candidates: bool = True


class CityBootstrapFromLocationResponse(BaseModel):
    city: CityResponse
    imported_count: int
    updated_count: int
    skipped_count: int
    stats: dict[str, Any] = Field(default_factory=dict)
    pois: list[PoiResponse]
