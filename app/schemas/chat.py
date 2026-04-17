from pydantic import BaseModel, Field

from app.schemas.poi import POI


class ChatSetupRequest(BaseModel):
    session_id: str
    user_id: int | None = None
    profile_context: str = Field(default="")
    lat: float | None = None
    lng: float | None = None


class ChatMessageRequest(BaseModel):
    session_id: str
    user_id: int | None = None
    message: str
    lat: float | None = None
    lng: float | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    pois: list[POI] = Field(default_factory=list)
    prompt_preview: str = Field(default="")
