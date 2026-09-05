from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


class LogLevelCount(BaseModel):
    level: str
    count: int


class LogItem(BaseModel):
    id: int
    level: str
    service: str
    environment: str
    event: str
    message: str | None
    trace_id: str | None
    user_id: int | None
    voice_session_id: int | None
    error_type: str | None
    error_code: str | None
    elapsed_ms: int | None
    context: dict[str, Any]
    created_at: datetime


class AdminLogPage(BaseModel):
    total: int
    limit: int
    offset: int
    levels: list[LogLevelCount]
    services: list[str]
    items: list[LogItem]


class AdminLogQuery(BaseModel):
    query: str = Field(default="", max_length=160)
    level: str | None = None
    service: str | None = None
    days: int = Field(default=7, ge=0, le=3650)
    limit: int = Field(default=100, ge=1, le=250)
    offset: int = Field(default=0, ge=0)


class AdminLogReader(Protocol):
    async def read(self, query: AdminLogQuery) -> AdminLogPage: ...


class AdminLogQueryService:
    def __init__(self, reader: AdminLogReader) -> None:
        self._reader = reader

    async def execute(self, query: AdminLogQuery) -> AdminLogPage:
        return await self._reader.read(query)
