"""/api/devices: el móvil de quien ha entrado, para mandarle avisos."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from locus_v2.api.auth import ClientAppDep, CurrentUserDep, SessionDep
from locus_v2.notifications.service import register_device

router = APIRouter(prefix="/api/devices", tags=["devices"])


class DeviceRequest(BaseModel):
    platform: Literal["ios", "android"]
    token: str = Field(min_length=8, max_length=255)


@router.post("", status_code=204)
async def add_device(
    body: DeviceRequest, session: SessionDep, app: ClientAppDep, current_user: CurrentUserDep
) -> None:
    await register_device(session, current_user, app, body.platform, body.token)
