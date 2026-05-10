from __future__ import annotations

import hashlib
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import Select, desc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import LedgerEntry, PriceSnapshot, TopUp, UsageEvent, User, Wallet
from app.db.session import session_scope
from app.services.openai_pricing_service import PricingRefreshError, openai_pricing_service


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

    def list_ledger_entries(self, user_id: int, limit: int = 50, offset: int = 0) -> list[LedgerEntry]:
        with session_scope() as db:
            stmt: Select[tuple[LedgerEntry]] = (
                select(LedgerEntry)
                .where(LedgerEntry.user_id == user_id)
                .order_by(desc(LedgerEntry.id))
                .offset(offset)
                .limit(limit)
            )
            return list(db.scalars(stmt).all())

    def get_usage_events_map(self, usage_ids: list[int]) -> dict[int, UsageEvent]:
        if not usage_ids:
            return {}
        with session_scope() as db:
            stmt = select(UsageEvent).where(UsageEvent.id.in_(usage_ids))
            events = list(db.scalars(stmt).all())
            return {event.id: event for event in events}

    def list_usage_events(self, user_id: int, limit: int = 100) -> list[UsageEvent]:
        with session_scope() as db:
            stmt: Select[tuple[UsageEvent]] = (
                select(UsageEvent)
                .where(UsageEvent.bill_to_user_id == user_id)
                .order_by(desc(UsageEvent.id))
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

    def _is_realtime_call_event(self, metadata: dict) -> bool:
        return (
            str(metadata.get("interaction_type") or "").strip() == "realtime_call"
            and str(metadata.get("call_id") or "").strip() != ""
        )

    def _list_realtime_call_usage_events(self, db: Session, *, user_id: int, call_id: str) -> list[UsageEvent]:
        stmt: Select[tuple[UsageEvent]] = (
            select(UsageEvent)
            .where(UsageEvent.bill_to_user_id == user_id)
            .where(UsageEvent.source == "call_room")
            .where(UsageEvent.interaction_type == "realtime_call")
            .where(UsageEvent.call_id == call_id)
            .order_by(UsageEvent.id.asc())
        )
        return list(db.scalars(stmt).all())

    def _microusd_to_provider_eur_cents(self, provider_cost_microusd: int) -> int:
        provider_cost_usd = Decimal(provider_cost_microusd) / Decimal(1_000_000)
        return int(
            (provider_cost_usd * Decimal(str(settings.billing_usd_to_eur)) * Decimal(100)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )

    def _microusd_to_charged_cents(self, provider_cost_microusd: int) -> int:
        provider_cost_usd = Decimal(provider_cost_microusd) / Decimal(1_000_000)
        return int(
            (provider_cost_usd * Decimal(str(settings.billing_usd_to_eur)) * Decimal(str(settings.billing_margin_multiplier)) * Decimal(100)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )

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

    def resolve_price(self, db: Session, *, provider: str, endpoint: str, model: str) -> PriceSnapshot:
        price = self.get_active_price(db, provider, endpoint, model)
        if provider != "openai":
            if price is None:
                raise BillingError(f"No hay tarifa registrada para {provider}:{endpoint}:{model}")
            return price

        try:
            return openai_pricing_service.get_or_refresh_snapshot(db, endpoint=endpoint, model=model)
        except PricingRefreshError as exc:
            if price is None:
                raise BillingError(f"No hay tarifa registrada para {provider}:{endpoint}:{model}") from exc
            return price

    def normalize_usage(self, usage: dict) -> dict[str, int]:
        input_details = usage.get("input_tokens_details") or usage.get("input_token_details") or {}
        output_details = usage.get("output_tokens_details") or usage.get("output_token_details") or {}

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        audio_input_tokens = int(input_details.get("audio_tokens", 0) or 0)
        audio_output_tokens = int(output_details.get("audio_tokens", 0) or 0)
        image_input_tokens = int(input_details.get("image_tokens", 0) or 0)
        cached_input_tokens = int(
            input_details.get(
                "cached_tokens",
                usage.get("cached_tokens", usage.get("cached_input_tokens", 0)),
            )
            or 0
        )
        text_input_tokens = int(input_details.get("text_tokens", max(input_tokens - audio_input_tokens - image_input_tokens, 0)) or 0)
        text_output_tokens = int(output_details.get("text_tokens", max(output_tokens - audio_output_tokens, 0)) or 0)
        reasoning_tokens = int(output_details.get("reasoning_tokens", 0) or 0)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "audio_input_tokens": audio_input_tokens,
            "audio_output_tokens": audio_output_tokens,
            "image_input_tokens": image_input_tokens,
            "text_input_tokens": text_input_tokens,
            "text_output_tokens": text_output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    def compute_charge(self, *, price: PriceSnapshot, usage: dict) -> tuple[int, int, int, dict[str, int]]:
        normalized = self.normalize_usage(usage)
        cached_tokens = normalized["cached_input_tokens"]
        text_input_tokens = normalized["text_input_tokens"]
        text_output_tokens = normalized["text_output_tokens"]
        audio_input_tokens = normalized["audio_input_tokens"]
        audio_output_tokens = normalized["audio_output_tokens"]
        image_input_tokens = normalized["image_input_tokens"]

        text_input_rate = self._decimal_or_fallback(price.text_input_per_million, price.input_per_million)
        text_cached_rate = self._decimal_or_fallback(price.text_cached_input_per_million, price.cached_input_per_million)
        text_output_rate = self._decimal_or_fallback(price.text_output_per_million, price.output_per_million)
        audio_input_rate = self._decimal_or_fallback(price.audio_input_per_million, Decimal("0"))
        audio_cached_rate = self._decimal_or_fallback(price.audio_cached_input_per_million, Decimal("0"))
        audio_output_rate = self._decimal_or_fallback(price.audio_output_per_million, Decimal("0"))
        image_input_rate = self._decimal_or_fallback(price.image_input_per_million, text_input_rate)
        image_cached_rate = self._decimal_or_fallback(price.image_cached_input_per_million, text_cached_rate)

        input_buckets = [
            {
                "tokens": audio_input_tokens,
                "fresh_rate": audio_input_rate,
                "cached_rate": audio_cached_rate,
            },
            {
                "tokens": image_input_tokens,
                "fresh_rate": image_input_rate,
                "cached_rate": image_cached_rate,
            },
            {
                "tokens": text_input_tokens,
                "fresh_rate": text_input_rate,
                "cached_rate": text_cached_rate,
            },
        ]

        remaining_cached = max(cached_tokens, 0)
        allocated_cached_total = 0
        input_cost_usd = Decimal("0")

        # If the provider only reports one cached_input_tokens bucket, allocate it first
        # to the modalities with the largest savings. This avoids overcharging when
        # realtime usage mixes text/audio cached context in one aggregate counter.
        input_buckets.sort(key=lambda item: item["fresh_rate"] - item["cached_rate"], reverse=True)
        for bucket in input_buckets:
            bucket_tokens = int(bucket["tokens"])
            if bucket_tokens <= 0:
                continue
            cached_here = min(remaining_cached, bucket_tokens)
            fresh_here = bucket_tokens - cached_here
            remaining_cached -= cached_here
            allocated_cached_total += cached_here
            input_cost_usd += (Decimal(fresh_here) / Decimal(1_000_000)) * bucket["fresh_rate"]
            input_cost_usd += (Decimal(cached_here) / Decimal(1_000_000)) * bucket["cached_rate"]

        provider_cost_usd = (
            input_cost_usd
            + (Decimal(text_output_tokens) / Decimal(1_000_000)) * text_output_rate
            + (Decimal(audio_output_tokens) / Decimal(1_000_000)) * audio_output_rate
        )
        provider_cost_microusd = int((provider_cost_usd * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        provider_cost_eur_cents = int(
            (provider_cost_usd * Decimal(str(settings.billing_usd_to_eur)) * Decimal(100)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )
        charged_eur = provider_cost_usd * Decimal(str(settings.billing_usd_to_eur)) * Decimal(str(settings.billing_margin_multiplier))
        charged_amount_cents = int((charged_eur * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        normalized["allocated_cached_input_tokens"] = allocated_cached_total
        normalized["unallocated_cached_input_tokens"] = max(remaining_cached, 0)
        return provider_cost_microusd, provider_cost_eur_cents, charged_amount_cents, normalized

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
            event_metadata = dict(metadata or {})
            dedupe_key = self._build_dedupe_key(provider=provider, endpoint=endpoint, model=model, response_id=response_id)
            if dedupe_key is not None:
                existing = db.scalar(select(UsageEvent).where(UsageEvent.dedupe_key == dedupe_key))
                if existing is not None:
                    wallet = self.get_wallet_or_raise(db, existing.bill_to_user_id or user_id)
                    return existing, wallet

            wallet = self.get_wallet_or_raise(db, user_id)
            price = self.resolve_price(db, provider=provider, endpoint=endpoint, model=model)
            provider_cost_microusd, provider_cost_eur_cents, charged_amount_cents, normalized = self.compute_charge(
                price=price,
                usage=usage,
            )

            if self._is_realtime_call_event(event_metadata):
                call_id = str(event_metadata.get("call_id") or "").strip()
                previous_events = self._list_realtime_call_usage_events(db, user_id=user_id, call_id=call_id)
                previous_provider_cost_microusd = sum(int(event.provider_cost_microusd or 0) for event in previous_events)
                previous_charged_cents = sum(int(event.charged_amount_cents or 0) for event in previous_events)
                total_provider_cost_microusd = previous_provider_cost_microusd + provider_cost_microusd
                total_call_charge_cents = self._microusd_to_charged_cents(total_provider_cost_microusd)
                if total_provider_cost_microusd > 0:
                    total_call_charge_cents = max(settings.billing_min_realtime_call_charge_cents, total_call_charge_cents)
                charged_amount_cents = max(total_call_charge_cents - previous_charged_cents, 0)
                event_metadata["call_charge_mode"] = "aggregated_by_call"
                event_metadata["call_total_provider_cost_microusd"] = total_provider_cost_microusd
                event_metadata["call_total_charge_cents"] = total_call_charge_cents
                event_metadata["call_previous_charge_cents"] = previous_charged_cents

            if charged_amount_cents > wallet.balance_cents:
                event_metadata["partial_charge"] = True
                event_metadata["requested_charge_cents"] = charged_amount_cents
                charged_amount_cents = wallet.balance_cents

            usage_event = UsageEvent(
                user_id=user_id,
                bill_to_user_id=user_id,
                session_id=session_id,
                provider=provider,
                endpoint=endpoint,
                model=model,
                interaction_type=str(event_metadata.get("interaction_type") or endpoint).strip(),
                source=str(event_metadata.get("source") or endpoint).strip(),
                call_id=str(event_metadata.get("call_id") or "").strip() or None,
                response_id=response_id,
                dedupe_key=dedupe_key,
                price_snapshot_id=price.id,
                input_tokens=normalized["input_tokens"],
                cached_input_tokens=normalized["cached_input_tokens"],
                output_tokens=normalized["output_tokens"],
                reasoning_tokens=normalized["reasoning_tokens"],
                audio_input_tokens=normalized["audio_input_tokens"],
                audio_output_tokens=normalized["audio_output_tokens"],
                image_input_tokens=normalized["image_input_tokens"],
                provider_cost_microusd=provider_cost_microusd,
                provider_cost_eur_cents=provider_cost_eur_cents,
                charged_amount_cents=charged_amount_cents,
                gross_margin_cents=charged_amount_cents - provider_cost_eur_cents,
                currency="EUR",
                margin_multiplier=Decimal(str(settings.billing_margin_multiplier)),
                status="partial" if event_metadata.get("partial_charge") else "charged",
                metadata_json=event_metadata,
            )
            db.add(usage_event)
            db.flush()

            if charged_amount_cents > 0:
                ledger_entry = self.add_ledger_entry(
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
                usage_event.metadata_json = {
                    **usage_event.metadata_json,
                    "ledger_entry_id": ledger_entry.id,
                }
                db.flush()
            return usage_event, wallet

    @staticmethod
    def _build_dedupe_key(*, provider: str, endpoint: str, model: str, response_id: str) -> str | None:
        clean_response_id = response_id.strip()
        if not clean_response_id:
            return None
        raw = f"{provider}:{endpoint}:{model}:{clean_response_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _decimal_or_fallback(primary: Decimal | None, fallback: Decimal | None) -> Decimal:
        primary_decimal = Decimal(str(primary or 0))
        if primary_decimal > 0:
            return primary_decimal
        return Decimal(str(fallback or 0))


billing_service = BillingService()
