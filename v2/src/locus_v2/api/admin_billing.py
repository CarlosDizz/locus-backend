from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.api.admin_auth import require_admin
from locus_v2.billing.application.admin_dashboard import (
    AdminBillingDashboard,
    AdminBillingDashboardService,
)
from locus_v2.billing.infrastructure.sqlalchemy_dashboard import (
    SqlAlchemyBillingDashboardReader,
)
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(
    prefix="/admin/v2/billing",
    tags=["admin-billing"],
    dependencies=[Depends(require_admin)],
)
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("", response_model=AdminBillingDashboard)
async def billing_dashboard(
    session: SessionDep,
    days: int = Query(default=0, ge=0, le=3650),
) -> AdminBillingDashboard:
    service = AdminBillingDashboardService(SqlAlchemyBillingDashboardReader(session))
    return await service.execute(days=days)
