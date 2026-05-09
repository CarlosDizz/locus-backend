from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


CallStatus = str


class CallParticipant(BaseModel):
    user_id: int
    display_name: str
    avatar_url: str = ""
    is_host: bool = False
    connected: bool = True
    joined_at: datetime


class UICapabilities(BaseModel):
    can_talk: bool = False
    can_text: bool = False
    can_image: bool = False
    can_interrupt: bool = False
    reason: str = ""


class CallSnapshot(BaseModel):
    call_id: str
    join_code: str
    host_user_id: int
    host_display_name: str
    host_session_id: str
    poi_id: int | str | None = None
    poi_name: str = ""
    status: CallStatus
    speaker_user_id: int | None = None
    max_members: int = 10
    member_count: int = 0
    participants: list[CallParticipant] = Field(default_factory=list)
    host_grace_deadline: datetime | None = None
    expires_at: datetime
    log: list[dict] = Field(default_factory=list)


class CallCreateRequest(BaseModel):
    session_id: str
    poi_id: int | str | None = None
    poi_name: str = ""


class CallCreateResponse(BaseModel):
    call: CallSnapshot
    join_token: str


class CallJoinTokenResponse(BaseModel):
    call: CallSnapshot
    join_token: str


class CallActionResponse(BaseModel):
    call: CallSnapshot

