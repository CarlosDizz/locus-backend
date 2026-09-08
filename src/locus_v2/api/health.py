from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.infrastructure.database.session import get_session

router = APIRouter(tags=["system"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]


class HealthResponse(BaseModel):
    status: str
    service: str
    database: str
    version: str


@router.get("/health/live", response_model=HealthResponse)
async def live_health() -> HealthResponse:
    return HealthResponse(status="ok", service="api", database="not_checked", version="0.1.0")


@router.get("/health/ready", response_model=HealthResponse)
async def ready_health(session: SessionDep) -> HealthResponse:
    await session.execute(text("SELECT 1"))
    return HealthResponse(status="ok", service="api", database="ok", version="0.1.0")


@router.get("/health", response_model=HealthResponse)
async def simple_health() -> HealthResponse:
    return await live_health()
