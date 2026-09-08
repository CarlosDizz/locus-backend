from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import Select, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.ai.models import AIModel, AIProvider
from locus_v2.billing.application.admin_dashboard import (
    AdminBillingDashboard,
    BillingDailyPoint,
    BillingLedgerItem,
    BillingModelBreakdown,
    BillingTotals,
    BillingUsageItem,
)
from locus_v2.billing.models import LedgerEntry, UsageEvent, UsageStatus
from locus_v2.identity.models import User
from locus_v2.shared.clock import utc_now


class SqlAlchemyBillingDashboardReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read(self, *, days: int) -> AdminBillingDashboard:
        since = self._since(days)
        return AdminBillingDashboard(
            period_days=days,
            totals=await self._totals(since),
            daily=await self._daily(since),
            by_model=await self._by_model(since),
            recent_usage=await self._recent_usage(since),
            recent_ledger=await self._recent_ledger(since),
        )

    async def _totals(self, since: datetime | None) -> BillingTotals:
        statement = select(
            func.count(UsageEvent.id),
            func.count(func.distinct(UsageEvent.voice_session_id)),
            func.count(func.distinct(UsageEvent.user_id)),
            func.coalesce(func.sum(UsageEvent.provider_cost_eur_cents), 0),
            func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0),
            func.coalesce(func.sum(UsageEvent.gross_margin_cents), 0),
            func.coalesce(
                func.sum(case((UsageEvent.status == UsageStatus.PENDING, 1), else_=0)), 0
            ),
            func.coalesce(
                func.sum(case((UsageEvent.status == UsageStatus.FAILED, 1), else_=0)), 0
            ),
        )
        row = (await self._session.execute(self._filtered(statement, since))).one()
        return BillingTotals(
            usage_events=int(row[0]),
            voice_sessions=int(row[1]),
            active_users=int(row[2]),
            provider_cost_eur_cents=int(row[3]),
            charged_amount_cents=int(row[4]),
            gross_margin_cents=int(row[5]),
            pending_events=int(row[6]),
            failed_events=int(row[7]),
        )

    async def _daily(self, since: datetime | None) -> list[BillingDailyPoint]:
        statement = (
            select(
                func.date(UsageEvent.created_at),
                func.count(UsageEvent.id),
                func.coalesce(func.sum(UsageEvent.provider_cost_eur_cents), 0),
                func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0),
            )
            .group_by(func.date(UsageEvent.created_at))
            .order_by(func.date(UsageEvent.created_at))
        )
        rows = (await self._session.execute(self._filtered(statement, since))).all()
        return [
            BillingDailyPoint(
                day=self._as_date(day),
                usage_events=int(events),
                provider_cost_eur_cents=int(cost),
                charged_amount_cents=int(charged),
            )
            for day, events, cost, charged in rows
        ]

    async def _by_model(self, since: datetime | None) -> list[BillingModelBreakdown]:
        statement = (
            select(
                AIProvider.code,
                AIModel.display_name,
                func.count(UsageEvent.id),
                func.coalesce(func.sum(UsageEvent.provider_cost_eur_cents), 0),
                func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0),
                func.coalesce(func.sum(UsageEvent.gross_margin_cents), 0),
            )
            .join(AIProvider, AIProvider.id == UsageEvent.provider_id)
            .join(AIModel, AIModel.id == UsageEvent.model_id)
            .group_by(AIProvider.code, AIModel.display_name)
            .order_by(func.sum(UsageEvent.charged_amount_cents).desc())
        )
        rows = (await self._session.execute(self._filtered(statement, since))).all()
        return [
            BillingModelBreakdown(
                provider=provider,
                model=model,
                usage_events=int(events),
                provider_cost_eur_cents=int(cost),
                charged_amount_cents=int(charged),
                gross_margin_cents=int(margin),
            )
            for provider, model, events, cost, charged, margin in rows
        ]

    async def _recent_usage(self, since: datetime | None) -> list[BillingUsageItem]:
        statement = (
            select(UsageEvent, User.email, AIProvider.code, AIModel.display_name)
            .outerjoin(User, User.id == UsageEvent.user_id)
            .join(AIProvider, AIProvider.id == UsageEvent.provider_id)
            .join(AIModel, AIModel.id == UsageEvent.model_id)
            .order_by(UsageEvent.created_at.desc())
            .limit(50)
        )
        rows = (await self._session.execute(self._filtered(statement, since))).all()
        return [
            BillingUsageItem(
                id=event.public_id,
                user_email=email,
                provider=provider,
                model=model,
                interaction_type=event.interaction_type,
                status=event.status,
                provider_cost_eur_cents=event.provider_cost_eur_cents,
                charged_amount_cents=event.charged_amount_cents,
                gross_margin_cents=event.gross_margin_cents,
                text_tokens=(
                    event.text_input_tokens
                    + event.cached_text_input_tokens
                    + event.text_output_tokens
                ),
                audio_tokens=(
                    event.audio_input_tokens
                    + event.cached_audio_input_tokens
                    + event.audio_output_tokens
                ),
                created_at=event.created_at,
            )
            for event, email, provider, model in rows
        ]

    async def _recent_ledger(self, since: datetime | None) -> list[BillingLedgerItem]:
        statement = (
            select(LedgerEntry, User.email)
            .join(User, User.id == LedgerEntry.user_id)
            .order_by(LedgerEntry.created_at.desc())
            .limit(40)
        )
        if since is not None:
            statement = statement.where(LedgerEntry.created_at >= since)
        rows = (await self._session.execute(statement)).all()
        return [
            BillingLedgerItem(
                id=entry.public_id,
                user_email=email,
                kind=entry.kind,
                description=entry.description,
                amount_cents=entry.amount_cents,
                balance_after_cents=entry.balance_after_cents,
                created_at=entry.created_at,
            )
            for entry, email in rows
        ]

    @staticmethod
    def _filtered(statement: Select[Any], since: datetime | None) -> Select[Any]:
        if since is None:
            return statement
        return statement.where(UsageEvent.created_at >= since)

    @staticmethod
    def _since(days: int) -> datetime | None:
        return utc_now() - timedelta(days=days) if days else None

    @staticmethod
    def _as_date(value: date | str) -> date:
        return value if isinstance(value, date) else date.fromisoformat(value)
