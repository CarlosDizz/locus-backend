"""Create the wallet in the same transaction as the new identity."""

from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.billing.models import LedgerEntry, LedgerEntryKind, Wallet


async def create_signup_wallet(session: AsyncSession, user_id: int, bonus_cents: int) -> Wallet:
    wallet = Wallet(user_id=user_id, currency="EUR", balance_cents=bonus_cents)
    session.add(wallet)
    await session.flush()
    if bonus_cents:
        session.add(LedgerEntry(
            user_id=user_id,
            wallet_id=wallet.id,
            kind=LedgerEntryKind.CREDIT,
            amount_cents=bonus_cents,
            currency="EUR",
            balance_after_cents=bonus_cents,
            description="Bono de bienvenida",
            reference_type="signup_bonus",
            reference_id=str(user_id),
            metadata_json={"legacy_entry_type": "signup_bonus"},
            trace_id=f"signup:{user_id}",
        ))
    return wallet
