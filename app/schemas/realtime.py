from pydantic import BaseModel, Field


class RealtimeSessionRequest(BaseModel):
    session_id: str
    user_id: int | None = None
    active_poi_name: str | None = None
    visit_context: str = Field(default="")


class RealtimeSessionResponse(BaseModel):
    session_id: str
    model: str
    transport: str
    webrtc_api_url: str
    instructions: str
    modalities: list[str]
    tools: list[dict]
    status: str = "ready"


class RealtimeClientSecretResponse(BaseModel):
    session_id: str
    model: str
    client_secret: str
    expires_at: int
    webrtc_api_url: str
    instructions: str
    voice: str
    modalities: list[str]
    tools: list[dict]


class RealtimeToolRequest(BaseModel):
    session_id: str
    tool_name: str
    arguments: dict


class RealtimePhotoInsightRequest(BaseModel):
    session_id: str
    image_data_url: str
    file_name: str | None = None


class RealtimePhotoInsightResponse(BaseModel):
    text: str
