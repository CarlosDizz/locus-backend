from pydantic import BaseModel, Field


class POI(BaseModel):
    id: str = Field(default="")
    name: str
    lat: float
    lng: float
    poi_type_code: str = Field(default="")
    description: str = Field(default="")
    summary: str = Field(default="")
    source_of_truth: str = Field(default="catalog")
    is_ephemeral: bool = Field(default=False)
    google_place_id: str = Field(default="")
