from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from locus_v2.billing.models import LedgerEntry, UsageEvent, Wallet
from locus_v2.identity.models import User
from locus_v2.voice.models import VoiceSession


class AdminUserQueryService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_users(
        self, *, query: str = "", status: str | None = None, limit: int = 50, offset: int = 0
    ) -> dict:
        filters = []
        if query.strip():
            term = f"%{query.strip()}%"
            filters.append(or_(User.display_name.like(term), User.email.like(term)))
        if status:
            filters.append(User.status == status)

        total = await self.session.scalar(
            select(func.count(User.id)).where(*filters)
        ) or 0
        wallet_balance = (
            select(Wallet.balance_cents).where(Wallet.user_id == User.id).scalar_subquery()
        )
        session_count = (
            select(func.count(VoiceSession.id))
            .where(VoiceSession.user_id == User.id)
            .scalar_subquery()
        )
        charged_total = (
            select(func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0))
            .where(UsageEvent.user_id == User.id)
            .scalar_subquery()
        )
        rows = (
            await self.session.execute(
                select(User, wallet_balance, session_count, charged_total)
                .options(selectinload(User.roles))
                .where(*filters)
                .order_by(User.created_at.desc())
                .limit(min(limit, 100))
                .offset(max(offset, 0))
            )
        ).all()
        return {
            "items": [
                self._summary(user, balance, sessions, charged)
                for user, balance, sessions, charged in rows
            ],
            "total": total,
            "limit": min(limit, 100),
            "offset": max(offset, 0),
        }

    async def user_detail(self, user_id: int) -> dict | None:
        user = await self.session.scalar(
            select(User).options(selectinload(User.roles)).where(User.id == user_id)
        )
        if user is None:
            return None
        wallet = await self.session.scalar(select(Wallet).where(Wallet.user_id == user.id))
        total_sessions = await self.session.scalar(
            select(func.count(VoiceSession.id)).where(VoiceSession.user_id == user.id)
        ) or 0
        usage = (
            await self.session.execute(
                select(
                    func.count(UsageEvent.id),
                    func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0),
                    func.coalesce(func.sum(UsageEvent.provider_cost_eur_cents), 0),
                ).where(UsageEvent.user_id == user.id)
            )
        ).one()
        voice_sessions = list(
            (
                await self.session.scalars(
                    select(VoiceSession)
                    .where(VoiceSession.user_id == user.id)
                    .order_by(VoiceSession.created_at.desc())
                    .limit(8)
                )
            ).all()
        )
        ledger_entries = (
            list(
                (
                    await self.session.scalars(
                        select(LedgerEntry)
                        .where(LedgerEntry.wallet_id == wallet.id)
                        .order_by(LedgerEntry.created_at.desc())
                        .limit(8)
                    )
                ).all()
            )
            if wallet
            else []
        )
        return {
            **self._summary(
                user, wallet.balance_cents if wallet else None, total_sessions, usage[1]
            ),
            "public_id": user.public_id,
            "legacy_v1_id": user.legacy_v1_id,
            "provider_subject": user.provider_subject,
            "usage_events": usage[0],
            "provider_cost_eur_cents": usage[2],
            "recent_voice_sessions": [
                {
                    "id": item.public_id,
                    "status": item.status,
                    "locale": item.locale,
                    "context_type": item.context_type,
                    "started_at": item.started_at,
                    "ended_at": item.ended_at,
                }
                for item in voice_sessions
            ],
            "ledger_entries": [
                {
                    "id": item.public_id,
                    "kind": item.kind,
                    "amount_cents": item.amount_cents,
                    "balance_after_cents": item.balance_after_cents,
                    "description": item.description,
                    "created_at": item.created_at,
                }
                for item in ledger_entries
            ],
        }

    @staticmethod
    def _summary(user: User, balance: int | None, sessions: int, charged: int) -> dict:
        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url,
            "auth_provider": user.auth_provider,
            "locale": user.locale,
            "status": user.status,
            "roles": sorted(role.code for role in user.roles),
            "balance_cents": balance or 0,
            "voice_sessions": sessions,
            "charged_amount_cents": charged,
            "created_at": user.created_at,
        }
