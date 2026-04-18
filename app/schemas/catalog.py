from datetime import datetime

from pydantic import BaseModel, Field


class CityCreateRequest(BaseModel):
    name: str
    slug: str | None = None
    country_code: str = ""
    lat: float | None = None
    lng: float | None = None
    source: str = "manual"


class CityBootstrapRequest(BaseModel):
    name: str
    country_code: str = ""


class CityResponse(BaseModel):
    id: int
    slug: str
    name: str
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


class PoiCreateRequest(BaseModel):
    city_id: int | None = None
    poi_type_id: int | None = None
    poi_type_code: str | None = None
    slug: str | None = None
    name: str
    lat: float | None = None
    lng: float | None = None
    short_description: str = ""
    long_description: str = ""
    source_of_truth: str = "manual"
    wikidata_id: str = ""
    wikipedia_title: str = ""
    google_place_id: str = ""
    is_active: bool = True
    metadata: dict = Field(default_factory=dict)


class PoiUpdateRequest(BaseModel):
    city_id: int | None = None
    poi_type_id: int | None = None
    poi_type_code: str | None = None
    slug: str | None = None
    name: str | None = None
    lat: float | None = None
    lng: float | None = None
    short_description: str | None = None
    long_description: str | None = None
    source_of_truth: str | None = None
    wikidata_id: str | None = None
    wikipedia_title: str | None = None
    google_place_id: str | None = None
    is_active: bool | None = None
    metadata: dict = Field(default_factory=dict)


class PoiResponse(BaseModel):
    id: int
    city_id: int | None = None
    poi_type_id: int | None = None
    poi_type_code: str | None = None
    poi_type_name: str | None = None
    slug: str
    name: str
    lat: float | None = None
    lng: float | None = None
    short_description: str
    long_description: str
    source_of_truth: str
    wikidata_id: str
    wikipedia_title: str
    google_place_id: str
    is_active: bool
    metadata: dict
    created_at: datetime
    updated_at: datetime


class PoiDocumentationResponse(BaseModel):
    poi: PoiResponse | None = None
    documentation: dict
    resolved_from_catalog: bool = False


class CityPoiImportRequest(BaseModel):
    radius_km: float = Field(default=8.0, gt=0, le=50)
    limit: int = Field(default=40, gt=0, le=150)
    use_ai_candidates: bool = True


class CityPoiImportResponse(BaseModel):
    city_id: int
    city_name: str
    imported_count: int
    updated_count: int
    skipped_count: int
    stats: dict = Field(default_factory=dict)
    pois: list[PoiResponse]
