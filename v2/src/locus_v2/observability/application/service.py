from datetime import timedelta
from typing import Any

import structlog

from locus_v2.observability.application.ports import EventLogRepository
from locus_v2.observability.domain import LogEntry, LogLevel
from locus_v2.shared.clock import utc_now

logger = structlog.get_logger()


class LocusEventLogger:
    """Persists operational events without ever breaking the user request."""

    def __init__(
        self,
        repository: EventLogRepository,
        *,
        service: str,
        environment: str,
    ) -> None:
        self.repository = repository
        self.service = service
        self.environment = environment

    async def write(
        self,
        level: LogLevel,
        event: str,
        *,
        message: str | None = None,
        trace_id: str | None = None,
        user_id: int | None = None,
        voice_session_id: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        elapsed_ms: float | int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        entry = LogEntry(
            level=level,
            service=self.service,
            event=event,
            environment=self.environment,
            message=message,
            trace_id=trace_id,
            user_id=user_id,
            voice_session_id=voice_session_id,
            error_type=error_type,
            error_code=error_code,
            elapsed_ms=round(elapsed_ms) if elapsed_ms is not None else None,
            context=context or {},
        )
        try:
            await self.repository.append(entry)
        except Exception:
            logger.exception(
                "locus_event_log_persistence_failed",
                source_event=event,
                source_service=self.service,
            )

    async def purge(self, retention_days: int) -> int:
        cutoff = utc_now() - timedelta(days=max(retention_days, 1))
        return await self.repository.delete_before(cutoff)
