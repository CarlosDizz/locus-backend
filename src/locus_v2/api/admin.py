from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2 import __version__
from locus_v2.admin.application.dto import AdminOverview, BuildInfo
from locus_v2.admin.application.service import AdminOverviewService
from locus_v2.admin.infrastructure.sqlalchemy_overview import SqlAlchemyOverviewReader
from locus_v2.api.admin_auth import require_admin
from locus_v2.config import Settings, get_settings
from locus_v2.infrastructure.database.session import get_session
from locus_v2.voice.providers.factory import build_provider_registry

router = APIRouter(prefix="/admin/v2", tags=["admin"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/overview", response_model=AdminOverview, dependencies=[Depends(require_admin)])
async def overview(session: SessionDep, settings: SettingsDep) -> AdminOverview:
    registry = build_provider_registry(settings)
    service = AdminOverviewService(SqlAlchemyOverviewReader(session))
    return await service.execute(
        environment=settings.env,
        build=BuildInfo(
            version=__version__,
            commit=settings.build_sha,
            deployed_at=settings.build_time,
        ),
        registered_adapters=registry.available(),
    )
