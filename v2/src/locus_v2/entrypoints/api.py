from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from locus_v2.api.admin import router as admin_router
from locus_v2.api.admin_audit import router as admin_audit_router
from locus_v2.api.admin_auth import router as admin_auth_router
from locus_v2.api.admin_billing import router as admin_billing_router
from locus_v2.api.admin_catalog import router as admin_catalog_router
from locus_v2.api.admin_configuration import router as admin_configuration_router
from locus_v2.api.admin_logs import router as admin_logs_router
from locus_v2.api.admin_users import router as admin_users_router
from locus_v2.api.app_info import router as app_info_router
from locus_v2.api.auth import router as mobile_auth_router
from locus_v2.api.billing import router as mobile_billing_router
from locus_v2.api.catalog import router as catalog_router
from locus_v2.api.chat import router as chat_router
from locus_v2.api.health import router as health_router
from locus_v2.api.legal import router as legal_router
from locus_v2.api.sessions import router as sessions_router
from locus_v2.config import get_settings
from locus_v2.infrastructure.database import get_database
from locus_v2.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await get_database().close()


app = FastAPI(
    title="Locus Backend V2",
    version="0.1.0",
    docs_url="/api/v2/docs",
    openapi_url="/api/v2/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/api/v2")
app.include_router(admin_router)
app.include_router(admin_audit_router)
app.include_router(admin_auth_router)
app.include_router(admin_billing_router)
app.include_router(admin_catalog_router)
app.include_router(admin_configuration_router)
app.include_router(admin_logs_router)
app.include_router(admin_users_router)
app.include_router(app_info_router)
app.include_router(legal_router)
app.include_router(mobile_auth_router)
app.include_router(mobile_billing_router)
app.include_router(catalog_router)
app.include_router(chat_router)
app.include_router(sessions_router)
