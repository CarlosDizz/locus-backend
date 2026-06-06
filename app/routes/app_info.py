from fastapi import APIRouter

from app.config import settings


router = APIRouter(prefix="/api/app", tags=["app"])


@router.get("/version")
async def app_version() -> dict:
    return {
        "android": {
            "latest_version_code": settings.app_android_latest_version_code,
            "update_url": settings.app_android_update_url,
        },
        "ios": {
            "latest_build": settings.app_ios_latest_build,
            "update_url": settings.app_ios_update_url,
        },
    }
