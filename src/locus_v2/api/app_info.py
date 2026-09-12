"""V1-compatible app-version facade, ported from app/routes/app_info.py.

Lets the Ionic app show a "please update" prompt when its build is older
than the configured minimum, without a redeploy — just an env var change.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from locus_v2.config import Settings, get_settings

router = APIRouter(prefix="/api/app", tags=["app"])
SettingsDep = Annotated[Settings, Depends(get_settings)]


class AndroidVersionInfo(BaseModel):
    latest_version_code: int
    update_url: str


class IosVersionInfo(BaseModel):
    latest_build: int
    update_url: str


class AppVersionResponse(BaseModel):
    android: AndroidVersionInfo
    ios: IosVersionInfo
    # Lo que se le regala a quien se registra. Viaja desde el servidor porque la
    # pantalla de login lo anuncia, y anunciarlo a fuego significa que el dia que
    # se cambie el importe la pagina de login mienta. Aqui basta con cambiar
    # LOCUS_BILLING_SIGNUP_BONUS_CENTS y reiniciar.
    signup_bonus_cents: int = 0


@router.get("/version", response_model=AppVersionResponse)
async def app_version(settings: SettingsDep) -> AppVersionResponse:
    return AppVersionResponse(
        android=AndroidVersionInfo(
            latest_version_code=settings.app_android_latest_version_code,
            update_url=settings.android_update_url(),
        ),
        ios=IosVersionInfo(
            latest_build=settings.app_ios_latest_build,
            update_url=settings.app_ios_update_url,
        ),
        signup_bonus_cents=settings.billing_signup_bonus_cents,
    )
