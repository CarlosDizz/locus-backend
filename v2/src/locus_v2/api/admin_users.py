from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.admin_auth import require_admin
from locus_v2.identity.application.admin_users import AdminUserQueryService
from locus_v2.identity.models import User, UserStatus
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(prefix="/admin/v2/users", tags=["admin-users"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[User, Depends(require_admin)]


@router.get("")
async def list_users(
    session: SessionDep,
    _admin: AdminDep,
    q: str = Query(default="", max_length=160),
    user_status: UserStatus | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return await AdminUserQueryService(session).list_users(
        query=q, status=user_status, limit=limit, offset=offset
    )


@router.get("/{user_id}")
async def user_detail(user_id: int, session: SessionDep, _admin: AdminDep) -> dict[str, Any]:
    detail = await AdminUserQueryService(session).user_detail(user_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return detail
