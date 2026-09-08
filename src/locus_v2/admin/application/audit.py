"""Read-side for the admin audit trail.

`AdminAuditEvent` rows are already written by `AdminConfigurationService`
(`admin/application/configuration.py`) on every model toggle, prompt
version create/publish, and routing change. This module only adds a way to
read them back, mirroring `observability.application.admin_logs` (Registros).
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from locus_v2.shared.clock import UtcDatetime


class AuditActionCount(BaseModel):
    action: str
    count: int


class AuditItem(BaseModel):
    id: int
    actor_user_id: int
    actor_name: str
    actor_email: str
    action: str
    resource_type: str
    resource_id: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    trace_id: str
    created_at: UtcDatetime


class AdminAuditPage(BaseModel):
    total: int
    limit: int
    offset: int
    actions: list[AuditActionCount]
    resource_types: list[str]
    items: list[AuditItem]


class AdminAuditQuery(BaseModel):
    query: str = Field(default="", max_length=160)
    action: str | None = None
    resource_type: str | None = None
    days: int = Field(default=30, ge=0, le=3650)
    limit: int = Field(default=100, ge=1, le=250)
    offset: int = Field(default=0, ge=0)


class AdminAuditReader(Protocol):
    async def read(self, query: AdminAuditQuery) -> AdminAuditPage: ...


class AdminAuditQueryService:
    def __init__(self, reader: AdminAuditReader) -> None:
        self._reader = reader

    async def execute(self, query: AdminAuditQuery) -> AdminAuditPage:
        return await self._reader.read(query)
