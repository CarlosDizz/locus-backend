from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.admin_auth import require_admin
from locus_v2.infrastructure.database.session import get_session
from locus_v2.observability.application.admin_logs import (
    AdminLogPage,
    AdminLogQuery,
    AdminLogQueryService,
)
from locus_v2.observability.infrastructure.sqlalchemy_reader import SQLAlchemyAdminLogReader

router = APIRouter(
    prefix="/admin/v2/logs",
    tags=["admin-logs"],
    dependencies=[Depends(require_admin)],
)
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("", response_model=AdminLogPage)
async def list_logs(
    session: SessionDep,
    q: str = Query(default="", max_length=160),
    level: str | None = Query(default=None, max_length=10),
    service: str | None = Query(default=None, max_length=40),
    days: int = Query(default=7, ge=0, le=3650),
    limit: int = Query(default=100, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
) -> AdminLogPage:
    query = AdminLogQuery(
        query=q,
        level=level,
        service=service,
        days=days,
        limit=limit,
        offset=offset,
    )
    return await AdminLogQueryService(SQLAlchemyAdminLogReader(session)).execute(query)
