from dataclasses import dataclass
from typing import cast

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.ai.models import AIModel, AIProvider, ProviderPriceSnapshot
from locus_v2.billing.application.processor import BillingDecisionEngine
from locus_v2.billing.models import LedgerEntry, LedgerEntryKind, UsageEvent, UsageStatus, Wallet
from locus_v2.billing.pricing import NormalizedUsage, PriceCard
from locus_v2.shared.clock import utc_now


@dataclass(frozen=True)
class ProcessedUsage:
    event_id: int
    trace_id: str
    user_id: int
    voice_session_id: int | None
    provider: str
    model: str
    provider_cost_microusd: int
    charged_amount_cents: int
    wallet_balance_cents: int
    partial_charge: bool


class MissingBillingConfiguration(RuntimeError):
    def __init__(self, event_id: int, message: str) -> None:
        super().__init__(message)
        self.event_id = event_id


class SqlAlchemyUsageProcessor:
    def __init__(self, session: AsyncSession, decision_engine: BillingDecisionEngine) -> None:
        self.session = session
        self.decision_engine = decision_engine

    async def process_next(self) -> ProcessedUsage | None:
        row = (
            await self.session.execute(
                select(UsageEvent, AIProvider.code, AIModel.external_id)
                .join(AIProvider, AIProvider.id == UsageEvent.provider_id)
                .join(AIModel, AIModel.id == UsageEvent.model_id)
                .where(UsageEvent.status == UsageStatus.PENDING)
                .order_by(UsageEvent.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).first()
        if row is None:
            return None

        event, provider_code, model_code = row
        if event.user_id is None:
            raise MissingBillingConfiguration(
                event.id,
                f"Usage event {event.id} has no billable user",
            )

        snapshot = await self._price_for(event)
        if snapshot is None:
            raise MissingBillingConfiguration(
                event.id,
                f"No active price for {provider_code}:{model_code} "
                f"at {event.created_at.isoformat()}",
            )
        wallet = await self.session.scalar(
            select(Wallet).where(Wallet.user_id == event.user_id).with_for_update()
        )
        if wallet is None:
            raise MissingBillingConfiguration(event.id, f"User {event.user_id} has no wallet")

        aggregate = event.interaction_type == "realtime_call" and event.voice_session_id is not None
        previous_cost, previous_charge = (
            await self._previous_call_totals(event) if aggregate else (0, 0)
        )
        decision = self.decision_engine.decide(
            usage=NormalizedUsage(
                text_input_tokens=event.text_input_tokens,
                cached_text_input_tokens=event.cached_text_input_tokens,
                text_output_tokens=event.text_output_tokens,
                audio_input_tokens=event.audio_input_tokens,
                cached_audio_input_tokens=event.cached_audio_input_tokens,
                audio_output_tokens=event.audio_output_tokens,
                image_input_tokens=event.image_input_tokens,
                cached_image_input_tokens=event.cached_image_input_tokens,
                audio_input_milliseconds=event.audio_input_milliseconds,
                audio_output_milliseconds=event.audio_output_milliseconds,
                tool_calls=event.tool_calls,
                raw=event.raw_usage_json,
            ),
            price=PriceCard.from_json(snapshot.pricing_json),
            wallet_balance_cents=wallet.balance_cents,
            previous_provider_cost_microusd=previous_cost,
            previous_charged_cents=previous_charge,
            aggregate_by_call=aggregate,
        )

        event.price_snapshot_id = snapshot.id
        event.provider_cost_microusd = decision.provider_cost_microusd
        event.provider_cost_eur_cents = decision.provider_cost_eur_cents
        event.charged_amount_cents = decision.charged_amount_cents
        event.gross_margin_cents = decision.gross_margin_cents
        event.margin_multiplier = self.decision_engine.margin_multiplier
        event.currency = wallet.currency
        event.status = UsageStatus.CHARGED
        billing_metadata = {
            "aggregate_by_call": aggregate,
            "call_total_provider_cost_microusd": decision.call_total_provider_cost_microusd,
            "call_total_charge_cents": decision.call_total_charge_cents,
            "call_previous_charge_cents": previous_charge,
            "requested_charge_cents": decision.requested_charge_cents,
            "partial_charge": decision.partial_charge,
            "processed_at": utc_now().isoformat(),
        }
        event.raw_usage_json = {**event.raw_usage_json, "_locus_billing": billing_metadata}

        if decision.charged_amount_cents:
            wallet.balance_cents -= decision.charged_amount_cents
            self.session.add(
                LedgerEntry(
                    user_id=event.user_id,
                    wallet_id=wallet.id,
                    usage_event_id=event.id,
                    kind=LedgerEntryKind.CHARGE,
                    amount_cents=-decision.charged_amount_cents,
                    currency=wallet.currency,
                    provider_cost_eur_cents=decision.provider_cost_eur_cents,
                    gross_margin_cents=decision.gross_margin_cents,
                    balance_after_cents=wallet.balance_cents,
                    exchange_rate=self.decision_engine.usd_to_eur,
                    margin_multiplier=self.decision_engine.margin_multiplier,
                    description=f"Consumo {provider_code}:{model_code}",
                    reference_type="usage_event",
                    reference_id=str(event.id),
                    metadata_json=billing_metadata,
                    trace_id=event.trace_id,
                )
            )

        return ProcessedUsage(
            event_id=event.id,
            trace_id=event.trace_id,
            user_id=event.user_id,
            voice_session_id=event.voice_session_id,
            provider=provider_code,
            model=model_code,
            provider_cost_microusd=decision.provider_cost_microusd,
            charged_amount_cents=decision.charged_amount_cents,
            wallet_balance_cents=wallet.balance_cents,
            partial_charge=decision.partial_charge,
        )

    async def _price_for(self, event: UsageEvent) -> ProviderPriceSnapshot | None:
        return cast(
            ProviderPriceSnapshot | None,
            await self.session.scalar(
                select(ProviderPriceSnapshot)
                .where(
                    ProviderPriceSnapshot.provider_id == event.provider_id,
                    ProviderPriceSnapshot.model_id == event.model_id,
                    ProviderPriceSnapshot.active.is_(True),
                    ProviderPriceSnapshot.effective_from <= event.created_at,
                    or_(
                        ProviderPriceSnapshot.effective_to.is_(None),
                        ProviderPriceSnapshot.effective_to > event.created_at,
                    ),
                )
                .order_by(
                    ProviderPriceSnapshot.effective_from.desc(),
                    ProviderPriceSnapshot.id.desc(),
                )
                .limit(1)
            ),
        )

    async def _previous_call_totals(self, event: UsageEvent) -> tuple[int, int]:
        values = (
            await self.session.execute(
                select(
                    func.coalesce(func.sum(UsageEvent.provider_cost_microusd), 0),
                    func.coalesce(func.sum(UsageEvent.charged_amount_cents), 0),
                ).where(
                    UsageEvent.voice_session_id == event.voice_session_id,
                    UsageEvent.id != event.id,
                    UsageEvent.status == UsageStatus.CHARGED,
                )
            )
        ).one()
        return int(values[0]), int(values[1])
