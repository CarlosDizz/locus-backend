from dataclasses import dataclass
from decimal import Decimal

from locus_v2.billing.pricing import (
    BillingPolicy,
    NormalizedUsage,
    PriceCard,
    ProviderCostCalculator,
)


@dataclass(frozen=True)
class BillingDecision:
    provider_cost_microusd: int
    provider_cost_eur_cents: int
    requested_charge_cents: int
    charged_amount_cents: int
    gross_margin_cents: int
    call_total_provider_cost_microusd: int
    call_total_charge_cents: int
    partial_charge: bool


class BillingDecisionEngine:
    """Calculates one event while rounding realtime charges at call level."""

    def __init__(
        self,
        *,
        usd_to_eur: Decimal,
        margin_multiplier: Decimal,
        minimum_realtime_call_charge_cents: int,
    ) -> None:
        self.usd_to_eur = usd_to_eur
        self.margin_multiplier = margin_multiplier
        self.minimum_realtime_call_charge_cents = minimum_realtime_call_charge_cents
        self.costs = ProviderCostCalculator()
        self.policy = BillingPolicy()

    def decide(
        self,
        *,
        usage: NormalizedUsage,
        price: PriceCard,
        wallet_balance_cents: int,
        previous_provider_cost_microusd: int = 0,
        previous_charged_cents: int = 0,
        aggregate_by_call: bool = False,
    ) -> BillingDecision:
        event_cost = self.costs.calculate_microusd(usage, price)
        previous_cost = previous_provider_cost_microusd if aggregate_by_call else 0
        previous_charge = previous_charged_cents if aggregate_by_call else 0
        total_cost = previous_cost + event_cost

        total_quote = self.policy.quote(total_cost, self.usd_to_eur, self.margin_multiplier)
        previous_quote = self.policy.quote(previous_cost, self.usd_to_eur, self.margin_multiplier)
        total_charge = total_quote.charged_amount_cents
        if aggregate_by_call and total_cost > 0:
            total_charge = max(total_charge, self.minimum_realtime_call_charge_cents)

        provider_cost_eur_cents = max(
            total_quote.provider_cost_eur_cents - previous_quote.provider_cost_eur_cents,
            0,
        )
        requested_charge = max(total_charge - previous_charge, 0)
        charged = min(requested_charge, max(wallet_balance_cents, 0))
        return BillingDecision(
            provider_cost_microusd=event_cost,
            provider_cost_eur_cents=provider_cost_eur_cents,
            requested_charge_cents=requested_charge,
            charged_amount_cents=charged,
            gross_margin_cents=charged - provider_cost_eur_cents,
            call_total_provider_cost_microusd=total_cost,
            call_total_charge_cents=total_charge,
            partial_charge=charged < requested_charge,
        )
