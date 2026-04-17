from fastapi import APIRouter

from app.config import settings


router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "app": settings.app_name,
        "env": settings.app_env,
        "build": settings.app_build,
    }
