from pydantic import BaseModel, Field


class POI(BaseModel):
    id: str = Field(default="")
    name: str
    lat: float
    lng: float
    description: str = Field(default="")
    summary: str = Field(default="")
