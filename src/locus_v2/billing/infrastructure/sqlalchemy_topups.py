from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.billing.application.admin_topups import (
    AdminTopUpsDashboard,
    TopUpDailyPoint,
    TopUpItem,
    TopUpProviderBreakdown,
    TopUpTotals,
)
from locus_v2.billing.models import TopUp, Wallet
from locus_v2.billing.topup_catalogue import estimated_fee_cents, estimated_fees_for
from locus_v2.identity.models import User
from locus_v2.shared.clock import utc_now

# Solo el dinero que llegó a abonarse cuenta como ingreso. Lo demás se enseña
# aparte, que es justo lo interesante de lo demás.
COMPLETED = "completed"


class SqlAlchemyTopUpsReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def read(self, *, days: int, limit: int) -> AdminTopUpsDashboard:
        since = self._since(days)
        by_provider = await self._by_provider(since)
        return AdminTopUpsDashboard(
            period_days=days,
            totals=await self._totals(since, by_provider),
            daily=await self._daily(since),
            by_provider=by_provider,
            recent=await self._recent(since, limit),
        )

    async def _by_provider(self, since: datetime | None) -> list[TopUpProviderBreakdown]:
        statement = (
            select(
                TopUp.provider,
                func.count(TopUp.id),
                func.coalesce(func.sum(TopUp.amount_cents), 0),
            )
            .where(TopUp.status == COMPLETED)
            .group_by(TopUp.provider)
            .order_by(func.sum(TopUp.amount_cents).desc())
        )
        rows = (await self._session.execute(self._filtered(statement, since))).all()
        return [
            TopUpProviderBreakdown(
                provider=provider,
                top_ups=int(count),
                gross_cents=int(gross),
                estimated_fees_cents=estimated_fees_for(provider, int(gross), int(count)),
            )
            for provider, count, gross in rows
        ]

    async def _totals(
        self, since: datetime | None, by_provider: list[TopUpProviderBreakdown]
    ) -> TopUpTotals:
        completed = select(
            func.count(TopUp.id),
            func.count(func.distinct(TopUp.user_id)),
            func.coalesce(func.sum(TopUp.amount_cents), 0),
            func.coalesce(func.sum(TopUp.bonus_cents), 0),
        ).where(TopUp.status == COMPLETED)
        count, payers, gross, bonus = (
            await self._session.execute(self._filtered(completed, since))
        ).one()

        unsettled = select(
            func.count(TopUp.id),
            func.coalesce(func.sum(TopUp.amount_cents), 0),
        ).where(TopUp.status != COMPLETED)
        pending_count, pending_cents = (
            await self._session.execute(self._filtered(unsettled, since))
        ).one()

        # Sin filtro de fecha a propósito: el saldo vivo es lo que se debe hoy,
        # no lo que se debía durante una ventana.
        outstanding = int(
            await self._session.scalar(
                select(func.coalesce(func.sum(Wallet.balance_cents), 0))
            )
            or 0
        )

        fees = sum(item.estimated_fees_cents for item in by_provider)
        return TopUpTotals(
            top_ups=int(count),
            paying_users=int(payers),
            gross_cents=int(gross),
            estimated_fees_cents=fees,
            net_cents=int(gross) - fees,
            average_cents=round(int(gross) / int(count)) if count else 0,
            bonus_cents=int(bonus),
            pending_top_ups=int(pending_count),
            pending_cents=int(pending_cents),
            outstanding_balance_cents=outstanding,
        )

    async def _daily(self, since: datetime | None) -> list[TopUpDailyPoint]:
        statement = (
            select(
                func.date(TopUp.created_at),
                func.count(TopUp.id),
                func.coalesce(func.sum(TopUp.amount_cents), 0),
            )
            .where(TopUp.status == COMPLETED)
            .group_by(func.date(TopUp.created_at))
            .order_by(func.date(TopUp.created_at))
        )
        rows = (await self._session.execute(self._filtered(statement, since))).all()
        return [
            TopUpDailyPoint(
                day=self._as_date(day), top_ups=int(count), gross_cents=int(gross)
            )
            for day, count, gross in rows
        ]

    async def _recent(self, since: datetime | None, limit: int) -> list[TopUpItem]:
        # Las pendientes salen aquí junto a las buenas: son el caso que hay que
        # ver, no un detalle que esconder en otra pantalla.
        statement = (
            select(TopUp, User.email, User.display_name)
            .outerjoin(User, User.id == TopUp.user_id)
            .order_by(TopUp.created_at.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(self._filtered(statement, since))).all()
        return [
            TopUpItem(
                id=str(topup.id),
                user_email=email or "",
                user_name=name or "",
                amount_cents=topup.amount_cents,
                bonus_cents=topup.bonus_cents,
                estimated_fee_cents=(
                    estimated_fee_cents(topup.provider, topup.amount_cents)
                    if topup.status == COMPLETED
                    else 0
                ),
                provider=topup.provider,
                provider_reference=topup.provider_reference,
                status=topup.status,
                created_at=topup.created_at,
                completed_at=topup.completed_at,
            )
            for topup, email, name in rows
        ]

    @staticmethod
    def _filtered(statement: Select[Any], since: datetime | None) -> Select[Any]:
        if since is None:
            return statement
        return statement.where(TopUp.created_at >= since)

    @staticmethod
    def _since(days: int) -> datetime | None:
        return utc_now() - timedelta(days=days) if days else None

    @staticmethod
    def _as_date(value: date | str) -> date:
        return value if isinstance(value, date) else date.fromisoformat(value)
