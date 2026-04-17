from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import Select, desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import LedgerEntry, PriceSnapshot, TopUp, UsageEvent, User, Wallet
from app.db.session import session_scope


class BillingError(RuntimeError):
    pass


class BillingService:
    def get_wallet(self, user_id: int) -> Wallet | None:
        with session_scope() as db:
            return db.scalar(select(Wallet).where(Wallet.user_id == user_id))

    def get_wallet_or_raise(self, db: Session, user_id: int) -> Wallet:
        wallet = db.scalar(select(Wallet).where(Wallet.user_id == user_id))
        if wallet is None:
            raise BillingError("Wallet no encontrada")
        return wallet

    def add_ledger_entry(
        self,
        *,
        db: Session,
        wallet: Wallet,
        user_id: int,
        entry_type: str,
        amount_cents: int,
        description: str,
        reference_type: str,
        reference_id: str,
        metadata: dict | None = None,
    ) -> LedgerEntry:
        wallet.balance_cents += amount_cents
        entry = LedgerEntry(
            wallet_id=wallet.id,
            user_id=user_id,
            entry_type=entry_type,
            amount_cents=amount_cents,
            balance_after_cents=wallet.balance_cents,
            description=description,
            reference_type=reference_type,
            reference_id=reference_id,
            metadata_json=metadata or {},
        )
        db.add(entry)
        db.flush()
        return entry

    def ensure_user_can_consume(self, user_id: int) -> None:
        with session_scope() as db:
            wallet = self.get_wallet_or_raise(db, user_id)
            if wallet.balance_cents < settings.billing_min_reserve_cents:
                raise BillingError("Saldo insuficiente para iniciar una nueva interacción")

    def list_ledger_entries(self, user_id: int, limit: int = 50) -> list[LedgerEntry]:
        with session_scope() as db:
            stmt: Select[tuple[LedgerEntry]] = (
                select(LedgerEntry)
                .where(LedgerEntry.user_id == user_id)
                .order_by(desc(LedgerEntry.id))
                .limit(limit)
            )
            return list(db.scalars(stmt).all())

    def create_topup(
        self,
        *,
        user_id: int,
        amount_cents: int,
        provider: str = "manual",
        provider_reference: str = "",
        metadata: dict | None = None,
    ) -> TopUp:
        with session_scope() as db:
            wallet = self.get_wallet_or_raise(db, user_id)
            topup = TopUp(
                user_id=user_id,
                wallet_id=wallet.id,
                amount_cents=amount_cents,
                bonus_cents=0,
                provider=provider,
                provider_reference=provider_reference,
                status="completed",
                metadata_json=metadata or {},
                completed_at=datetime.utcnow(),
            )
            db.add(topup)
            db.flush()
            self.add_ledger_entry(
                db=db,
                wallet=wallet,
                user_id=user_id,
                entry_type="topup",
                amount_cents=amount_cents,
                description="Recarga de saldo",
                reference_type="top_up",
                reference_id=str(topup.id),
                metadata=metadata,
            )
            return topup

    def get_active_price(self, db: Session, provider: str, endpoint: str, model: str) -> PriceSnapshot | None:
        stmt = (
            select(PriceSnapshot)
            .where(PriceSnapshot.provider == provider)
            .where(PriceSnapshot.endpoint == endpoint)
            .where(PriceSnapshot.model == model)
            .order_by(desc(PriceSnapshot.active_from), desc(PriceSnapshot.id))
            .limit(1)
        )
        return db.scalar(stmt)

    def compute_charge(self, *, price: PriceSnapshot | None, usage: dict) -> tuple[int, int]:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cached_tokens = int(
            usage.get("input_tokens_details", {}).get(
                "cached_tokens",
                usage.get("cached_tokens", usage.get("cached_input_tokens", 0)),
            )
            or 0
        )

        if price is None:
            return 0, 0

        normal_input_tokens = max(input_tokens - cached_tokens, 0)
        provider_cost_usd = (
            (Decimal(normal_input_tokens) / Decimal(1_000_000)) * Decimal(price.input_per_million)
            + (Decimal(cached_tokens) / Decimal(1_000_000)) * Decimal(price.cached_input_per_million)
            + (Decimal(output_tokens) / Decimal(1_000_000)) * Decimal(price.output_per_million)
        )
        provider_cost_microusd = int((provider_cost_usd * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        charged_eur = provider_cost_usd * Decimal(str(settings.billing_usd_to_eur)) * Decimal(str(settings.billing_margin_multiplier))
        charged_amount_cents = int((charged_eur * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return provider_cost_microusd, charged_amount_cents

    def record_usage(
        self,
        *,
        user_id: int,
        session_id: str | None,
        provider: str,
        endpoint: str,
        model: str,
        response_id: str,
        usage: dict,
        metadata: dict | None = None,
    ) -> tuple[UsageEvent, Wallet]:
        with session_scope() as db:
            wallet = self.get_wallet_or_raise(db, user_id)
            price = self.get_active_price(db, provider, endpoint, model)
            provider_cost_microusd, charged_amount_cents = self.compute_charge(price=price, usage=usage)
            event_metadata = dict(metadata or {})
            if charged_amount_cents > wallet.balance_cents:
                event_metadata["partial_charge"] = True
                event_metadata["requested_charge_cents"] = charged_amount_cents
                charged_amount_cents = wallet.balance_cents

            usage_event = UsageEvent(
                user_id=user_id,
                session_id=session_id,
                provider=provider,
                endpoint=endpoint,
                model=model,
                response_id=response_id,
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                cached_input_tokens=int(
                    usage.get("input_tokens_details", {}).get(
                        "cached_tokens",
                        usage.get("cached_tokens", 0),
                    )
                    or 0
                ),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                reasoning_tokens=int(usage.get("output_tokens_details", {}).get("reasoning_tokens", 0) or 0),
                audio_input_tokens=int(usage.get("input_token_details", {}).get("audio_tokens", 0) or 0),
                audio_output_tokens=int(usage.get("output_token_details", {}).get("audio_tokens", 0) or 0),
                image_input_tokens=int(usage.get("input_token_details", {}).get("image_tokens", 0) or 0),
                provider_cost_microusd=provider_cost_microusd,
                charged_amount_cents=charged_amount_cents,
                currency="EUR",
                margin_multiplier=Decimal(str(settings.billing_margin_multiplier)),
                metadata_json=event_metadata,
            )
            db.add(usage_event)
            db.flush()

            if charged_amount_cents > 0:
                self.add_ledger_entry(
                    db=db,
                    wallet=wallet,
                    user_id=user_id,
                    entry_type="usage_charge",
                    amount_cents=-charged_amount_cents,
                    description=f"Consumo {provider}:{model}",
                    reference_type="usage_event",
                    reference_id=str(usage_event.id),
                    metadata=event_metadata,
                )
            return usage_event, wallet


billing_service = BillingService()
