import json
from datetime import datetime, timedelta

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.admin.application.audit import (
    AdminAuditPage,
    AdminAuditQuery,
    AuditActionCount,
    AuditItem,
)
from locus_v2.identity.models import AdminAuditEvent, User
from locus_v2.shared.clock import utc_now


class SqlAlchemyAdminAuditReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read(self, query: AdminAuditQuery) -> AdminAuditPage:
        since = utc_now() - timedelta(days=query.days) if query.days else None
        filters = self._filters(query, since)

        total_statement = select(func.count()).select_from(AdminAuditEvent).where(*filters)
        total = int(await self._session.scalar(total_statement) or 0)

        rows_statement = (
            select(AdminAuditEvent, User.display_name, User.email)
            .join(User, User.id == AdminAuditEvent.actor_user_id)
            .where(*filters)
            .order_by(AdminAuditEvent.created_at.desc())
            .limit(query.limit)
            .offset(query.offset)
        )
        rows = (await self._session.execute(rows_statement)).all()

        action_statement = (
            select(AdminAuditEvent.action, func.count(AdminAuditEvent.id))
            .where(*self._filters(query, since, include_action=False))
            .group_by(AdminAuditEvent.action)
        )
        action_rows = (await self._session.execute(action_statement)).all()

        resource_statement = (
            select(AdminAuditEvent.resource_type)
            .where(*self._filters(query, since, include_resource_type=False))
            .distinct()
            .order_by(AdminAuditEvent.resource_type)
        )
        resource_types = list((await self._session.scalars(resource_statement)).all())

        return AdminAuditPage(
            total=total,
            limit=query.limit,
            offset=query.offset,
            actions=[
                AuditActionCount(action=action, count=int(count)) for action, count in action_rows
            ],
            resource_types=resource_types,
            items=[self._item(row, name, email) for row, name, email in rows],
        )

    @staticmethod
    def _filters(
        query: AdminAuditQuery,
        since: datetime | None,
        *,
        include_action: bool = True,
        include_resource_type: bool = True,
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if since is not None:
            filters.append(AdminAuditEvent.created_at >= since)
        if include_action and query.action:
            filters.append(AdminAuditEvent.action == query.action)
        if include_resource_type and query.resource_type:
            filters.append(AdminAuditEvent.resource_type == query.resource_type)
        if query.query:
            pattern = f"%{query.query.strip()}%"
            filters.append(
                or_(
                    AdminAuditEvent.action.ilike(pattern),
                    AdminAuditEvent.resource_type.ilike(pattern),
                    AdminAuditEvent.resource_id.ilike(pattern),
                    AdminAuditEvent.trace_id.ilike(pattern),
                )
            )
        return filters

    @staticmethod
    def _item(row: AdminAuditEvent, actor_name: str, actor_email: str) -> AuditItem:
        return AuditItem(
            id=row.id,
            actor_user_id=row.actor_user_id,
            actor_name=actor_name,
            actor_email=actor_email,
            action=row.action,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            before=json.loads(row.before_json) if row.before_json else None,
            after=json.loads(row.after_json) if row.after_json else None,
            trace_id=row.trace_id,
            created_at=row.created_at,
        )
