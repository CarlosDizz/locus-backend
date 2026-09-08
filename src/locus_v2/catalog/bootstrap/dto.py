from pydantic import BaseModel


class BootstrapPoi(BaseModel):
    id: int
    public_id: str
    name: str
    slug: str
    type_code: str | None
    lat: float | None
    lng: float | None
    short_description: str
    source_of_truth: str
    created: bool


class BootstrapResult(BaseModel):
    city_id: int
    city_public_id: str
    city_name: str
    city_created: bool
    source: str
    imported_count: int
    updated_count: int
    pois: list[BootstrapPoi]
