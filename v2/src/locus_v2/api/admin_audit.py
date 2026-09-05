from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.admin.application.audit import (
    AdminAuditPage,
    AdminAuditQuery,
    AdminAuditQueryService,
)
from locus_v2.admin.infrastructure.sqlalchemy_audit import SqlAlchemyAdminAuditReader
from locus_v2.api.admin_auth import require_admin
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(
    prefix="/admin/v2/audit",
    tags=["admin-audit"],
    dependencies=[Depends(require_admin)],
)
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("", response_model=AdminAuditPage)
async def list_audit_events(
    session: SessionDep,
    q: str = Query(default="", max_length=160),
    action: str | None = Query(default=None, max_length=100),
    resource_type: str | None = Query(default=None, max_length=80),
    days: int = Query(default=30, ge=0, le=3650),
    limit: int = Query(default=100, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
) -> AdminAuditPage:
    query = AdminAuditQuery(
        query=q,
        action=action,
        resource_type=resource_type,
        days=days,
        limit=limit,
        offset=offset,
    )
    return await AdminAuditQueryService(SqlAlchemyAdminAuditReader(session)).execute(query)
