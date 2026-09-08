from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2 import __version__
from locus_v2.config import Settings, get_settings
from locus_v2.infrastructure.database.session import get_session

router = APIRouter(tags=["system"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class HealthResponse(BaseModel):
    status: str
    service: str
    database: str
    # The release number we maintain by hand, plus the commit the deploy put
    # here. Keep both: the version is what a human says, the commit is what is
    # actually running, and only the second one cannot be forgotten. This used
    # to be the literal string "0.1.0", which answered "which build is live?"
    # with the same value forever.
    version: str
    commit: str
    deployed_at: str


def _report(settings: Settings, *, database: str) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="api",
        database=database,
        version=__version__,
        commit=settings.build_sha,
        deployed_at=settings.build_time,
    )


@router.get("/health/live", response_model=HealthResponse)
async def live_health(settings: SettingsDep) -> HealthResponse:
    return _report(settings, database="not_checked")


@router.get("/health/ready", response_model=HealthResponse)
async def ready_health(session: SessionDep, settings: SettingsDep) -> HealthResponse:
    await session.execute(text("SELECT 1"))
    return _report(settings, database="ok")


@router.get("/health", response_model=HealthResponse)
async def simple_health(settings: SettingsDep) -> HealthResponse:
    return await live_health(settings)
