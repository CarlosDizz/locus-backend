from datetime import datetime, timedelta

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.observability.application.admin_logs import (
    AdminLogPage,
    AdminLogQuery,
    LogItem,
    LogLevelCount,
)
from locus_v2.observability.models import LocusLog
from locus_v2.shared.clock import utc_now


class SQLAlchemyAdminLogReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read(self, query: AdminLogQuery) -> AdminLogPage:
        since = utc_now() - timedelta(days=query.days) if query.days else None
        filters = self._filters(query, since)

        total_statement = select(func.count()).select_from(LocusLog).where(*filters)
        total = int(await self._session.scalar(total_statement) or 0)

        rows_statement = (
            select(LocusLog)
            .where(*filters)
            .order_by(LocusLog.created_at.desc())
            .limit(query.limit)
            .offset(query.offset)
        )
        rows = (await self._session.scalars(rows_statement)).all()

        level_statement = (
            select(LocusLog.level, func.count(LocusLog.id))
            .where(*self._filters(query, since, include_level=False))
            .group_by(LocusLog.level)
        )
        level_rows = (await self._session.execute(level_statement)).all()

        service_statement = (
            select(LocusLog.service)
            .where(*self._filters(query, since, include_service=False))
            .distinct()
            .order_by(LocusLog.service)
        )
        services = list((await self._session.scalars(service_statement)).all())

        return AdminLogPage(
            total=total,
            limit=query.limit,
            offset=query.offset,
            levels=[LogLevelCount(level=level, count=int(count)) for level, count in level_rows],
            services=services,
            items=[self._item(row) for row in rows],
        )

    @staticmethod
    def _filters(
        query: AdminLogQuery,
        since: datetime | None,
        *,
        include_level: bool = True,
        include_service: bool = True,
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if since is not None:
            filters.append(LocusLog.created_at >= since)
        if include_level and query.level:
            filters.append(LocusLog.level == query.level)
        if include_service and query.service:
            filters.append(LocusLog.service == query.service)
        if query.query:
            pattern = f"%{query.query.strip()}%"
            filters.append(
                or_(
                    LocusLog.event.ilike(pattern),
                    LocusLog.message.ilike(pattern),
                    LocusLog.trace_id.ilike(pattern),
                    LocusLog.error_code.ilike(pattern),
                )
            )
        return filters

    @staticmethod
    def _item(row: LocusLog) -> LogItem:
        return LogItem(
            id=row.id,
            level=row.level,
            service=row.service,
            environment=row.environment,
            event=row.event,
            message=row.message,
            trace_id=row.trace_id,
            user_id=row.user_id,
            voice_session_id=row.voice_session_id,
            error_type=row.error_type,
            error_code=row.error_code,
            elapsed_ms=row.elapsed_ms,
            context=row.context_json,
            created_at=row.created_at,
        )
