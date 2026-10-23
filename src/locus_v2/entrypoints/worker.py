import asyncio
import signal
from datetime import timedelta

import structlog
from redis.asyncio import Redis
from sqlalchemy import delete, or_, select

from locus_v2.billing.application.processor import BillingDecisionEngine
from locus_v2.billing.infrastructure.sqlalchemy_processor import (
    MissingBillingConfiguration,
    SqlAlchemyUsageProcessor,
)
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.config import get_settings
from locus_v2.identity.models import AdminSession, UserSession
from locus_v2.infrastructure.database import models as database_models  # noqa: F401
from locus_v2.infrastructure.database.session import get_database
from locus_v2.logging import configure_logging
from locus_v2.observability.application.service import LocusEventLogger
from locus_v2.observability.infrastructure.sqlalchemy_repository import (
    SQLAlchemyEventLogRepository,
)
from locus_v2.observability.models import LocusLog
from locus_v2.shared.clock import utc_now

# Cada cuanto se limpian los registros viejos. No hace falta afinar: lo que
# importa es que no crezcan sin limite en un disco de 20 GB.
PRUNE_EVERY_SECONDS = 6 * 3600
# Cuanto se guardan las sesiones de login ya caducadas o cerradas. Ya no sirven
# para entrar; se dejan un tiempo solo por si hace falta saber quien entro.
EXPIRED_SESSION_RETENTION_DAYS = 30


async def process_pending_usage(stop: asyncio.Event) -> None:
    settings = get_settings()
    database = get_database()
    event_logger = LocusEventLogger(
        SQLAlchemyEventLogRepository(database),
        service="billing-worker",
        environment=settings.env,
    )
    decision_engine = BillingDecisionEngine(
        usd_to_eur=settings.billing_usd_to_eur,
        margin_multiplier=settings.billing_margin_multiplier,
        minimum_realtime_call_charge_cents=settings.billing_min_realtime_call_charge_cents,
        free_apps=frozenset({"borea"}) if settings.borea_free else frozenset(),
    )
    logger = structlog.get_logger()

    while not stop.is_set():
        try:
            async with database.sessions() as session:
                processor = SqlAlchemyUsageProcessor(session, decision_engine)
                result = await processor.process_next()
                await session.commit()
        except MissingBillingConfiguration as error:
            async with database.sessions() as session:
                event = await session.scalar(
                    select(UsageEvent).where(UsageEvent.id == error.event_id).with_for_update()
                )
                if event is not None and event.status == UsageStatus.PENDING:
                    event.status = UsageStatus.FAILED
                    event.raw_usage_json = {
                        **event.raw_usage_json,
                        "_locus_billing": {"error": str(error)},
                    }
                    await session.commit()
            logger.error("billing_usage_failed", event_id=error.event_id, error=str(error))
            await event_logger.write(
                "error",
                "billing.usage.failed",
                message=str(error),
                error_type=type(error).__name__,
                context={"usage_event_id": error.event_id},
            )
            continue
        except Exception as error:
            logger.exception("billing_worker_iteration_failed")
            await event_logger.write(
                "error",
                "billing.worker.failed",
                message=str(error),
                error_type=type(error).__name__,
            )
            await _wait_or_stop(stop, settings.billing_worker_poll_seconds)
            continue

        if result is None:
            await _wait_or_stop(stop, settings.billing_worker_poll_seconds)
            continue

        logger.info(
            "billing_usage_charged",
            event_id=result.event_id,
            provider=result.provider,
            model=result.model,
            provider_cost_microusd=result.provider_cost_microusd,
            charged_amount_cents=result.charged_amount_cents,
            wallet_balance_cents=result.wallet_balance_cents,
            partial_charge=result.partial_charge,
        )
        await event_logger.write(
            "info",
            "billing.usage.charged",
            trace_id=result.trace_id,
            user_id=result.user_id,
            voice_session_id=result.voice_session_id,
            context={
                "usage_event_id": result.event_id,
                "provider": result.provider,
                "model": result.model,
                "provider_cost_microusd": result.provider_cost_microusd,
                "charged_amount_cents": result.charged_amount_cents,
                "wallet_balance_cents": result.wallet_balance_cents,
                "partial_charge": result.partial_charge,
            },
        )


async def prune_old_records(stop: asyncio.Event) -> None:
    """Aplica la retencion: logs de mas de event_log_retention_days y sesiones muertas.

    event_log_retention_days existia desde el principio pero nadie lo aplicaba,
    asi que locus_logs crecia sin limite. Las sesiones caducadas ya no dejan
    entrar a nadie (authenticate() mira expires_at y revoked_at); borrarlas es
    solo no acumular filas.
    """
    settings = get_settings()
    database = get_database()
    logger = structlog.get_logger()
    while not stop.is_set():
        now = utc_now()
        logs_before = now - timedelta(days=settings.event_log_retention_days)
        sessions_before = now - timedelta(days=EXPIRED_SESSION_RETENTION_DAYS)
        try:
            async with database.sessions() as session:
                logs = await session.execute(
                    delete(LocusLog).where(LocusLog.created_at < logs_before)
                )
                sesiones = 0
                for model in (UserSession, AdminSession):
                    result = await session.execute(
                        delete(model).where(
                            or_(
                                model.expires_at < sessions_before,
                                model.revoked_at < sessions_before,
                            )
                        )
                    )
                    sesiones += result.rowcount or 0
                await session.commit()
            logger.info(
                "worker_pruned_old_records",
                logs=logs.rowcount or 0,
                sessions=sesiones,
            )
        except Exception:  # noqa: BLE001 - la limpieza no puede tumbar el cobro
            logger.exception("worker_prune_failed")
        await _wait_or_stop(stop, PRUNE_EVERY_SECONDS)


async def _wait_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = structlog.get_logger()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    logger.info("worker_ready", environment=settings.env)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await asyncio.gather(process_pending_usage(stop), prune_old_records(stop))
    await redis.aclose()
    await get_database().close()


if __name__ == "__main__":
    asyncio.run(run())
