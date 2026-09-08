import asyncio
import signal

import structlog
from redis.asyncio import Redis
from sqlalchemy import select

from locus_v2.billing.application.processor import BillingDecisionEngine
from locus_v2.billing.infrastructure.sqlalchemy_processor import (
    MissingBillingConfiguration,
    SqlAlchemyUsageProcessor,
)
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.config import get_settings
from locus_v2.infrastructure.database import models as database_models  # noqa: F401
from locus_v2.infrastructure.database.session import get_database
from locus_v2.logging import configure_logging
from locus_v2.observability.application.service import LocusEventLogger
from locus_v2.observability.infrastructure.sqlalchemy_repository import (
    SQLAlchemyEventLogRepository,
)


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
    await process_pending_usage(stop)
    await redis.aclose()
    await get_database().close()


if __name__ == "__main__":
    asyncio.run(run())
