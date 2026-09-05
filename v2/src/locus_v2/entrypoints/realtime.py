from fastapi import FastAPI, WebSocket

from locus_v2.config import get_settings
from locus_v2.infrastructure.database.session import get_database
from locus_v2.logging import configure_logging
from locus_v2.observability import LocusEventLogger
from locus_v2.observability.infrastructure import SQLAlchemyEventLogRepository
from locus_v2.voice.auth import VoiceAuthenticationError, authenticate_voice_user
from locus_v2.voice.gateway import VoiceGateway
from locus_v2.voice.providers.factory import build_provider_registry

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Locus Voice Gateway V2", docs_url=None, redoc_url=None)


@app.get("/ws/v2/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "realtime", "version": "0.1.0"}


@app.websocket("/ws/v2/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    database = get_database()
    async with database.sessions() as session:
        try:
            user = await authenticate_voice_user(websocket, session, settings)
        except VoiceAuthenticationError:
            await websocket.close(code=4401, reason="Authentication required")
            return
        gateway = VoiceGateway(
            websocket=websocket,
            session=session,
            database=database,
            settings=settings,
            registry=build_provider_registry(settings),
            user=user,
            event_logger=LocusEventLogger(
                SQLAlchemyEventLogRepository(database),
                service="realtime",
                environment=settings.env,
            ),
        )
        await gateway.run()
