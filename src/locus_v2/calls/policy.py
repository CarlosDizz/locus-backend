from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.billing.models import Wallet
from locus_v2.calls.models import CallError, CreateCall
from locus_v2.catalog.models import Poi
from locus_v2.config import Settings
from locus_v2.identity.models import User, UserStatus
from locus_v2.infrastructure.database.session import Database
from locus_v2.sessions.models import MapSession
from locus_v2.shared.mobile_ids import mobile_id_clause


async def ensure_host_can_consume(database: Database, settings: Settings, user_id: int) -> None:
    # Use a fresh transaction: the billing worker can update the host wallet mid-call.
    async with database.sessions() as session:
        balance = await session.scalar(
            select(Wallet.balance_cents)
            .join(User, User.id == Wallet.user_id)
            .where(
                User.id == user_id,
                User.status == UserStatus.ACTIVE,
            )
        )
        if balance is None or balance < max(1, settings.billing_min_reserve_cents):
            raise CallError("Host balance is insufficient or account unavailable", 402)


async def resolve_context(session: AsyncSession, user: User, request: CreateCall) -> Poi:
    map_session = await session.get(MapSession, request.session_id.upper())
    if map_session is None or map_session.user_id != user.id:
        raise CallError("An owned map session is required", 403)
    value = request.poi_id
    if value is None:
        value = (map_session.active_poi_json or {}).get("id")
    if value is None:
        raise CallError("A catalog POI is required; ephemeral POIs are not supported", 422)
    clause = (
        mobile_id_clause(Poi, int(value)) if str(value).isdigit() else Poi.public_id == str(value)
    )
    poi = await session.scalar(select(Poi).where(clause, Poi.is_active.is_(True)))
    if poi is None:
        raise CallError("Active catalog POI not found", 404)
    return poi
