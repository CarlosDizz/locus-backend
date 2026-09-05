from datetime import datetime

from sqlalchemy import delete, func, select

from locus_v2.infrastructure.database.session import Database
from locus_v2.observability.domain import LogEntry
from locus_v2.observability.models import LocusLog


class SQLAlchemyEventLogRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def append(self, entry: LogEntry) -> None:
        async with self.database.sessions() as session:
            session.add(
                LocusLog(
                    level=entry.level,
                    service=entry.service,
                    environment=entry.environment,
                    event=entry.event,
                    message=entry.message,
                    trace_id=entry.trace_id,
                    user_id=entry.user_id,
                    voice_session_id=entry.voice_session_id,
                    error_type=entry.error_type,
                    error_code=entry.error_code,
                    elapsed_ms=entry.elapsed_ms,
                    context_json=entry.context,
                )
            )
            await session.commit()

    async def delete_before(self, cutoff: datetime) -> int:
        async with self.database.sessions() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(LocusLog)
                .where(LocusLog.created_at < cutoff)
            )
            await session.execute(delete(LocusLog).where(LocusLog.created_at < cutoff))
            await session.commit()
            return count or 0
