from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.auth import router as auth_router
from app.routes.billing import router as billing_router
from app.routes.catalog import router as catalog_router
from app.routes.chat import compat_router as chat_compat_router
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.realtime import router as realtime_router
from app.routes.sessions import router as sessions_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(billing_router)
    app.include_router(catalog_router)
    app.include_router(sessions_router)
    app.include_router(chat_router)
    app.include_router(realtime_router)
    app.include_router(chat_compat_router)
    return app


app = create_app()
