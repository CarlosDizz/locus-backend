from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

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
        exact_apps: frozenset[str] = frozenset({"borea"}),
        free_apps: frozenset[str] = frozenset(),
    ) -> None:
        self.usd_to_eur = usd_to_eur
        self.margin_multiplier = margin_multiplier
        self.minimum_realtime_call_charge_cents = minimum_realtime_call_charge_cents
        # Apps que cobran cada consumo exacto, sin redondear al céntimo: el saldo va
        # en céntimos y lo que no llega a uno se arrastra (Wallet.carry_microeur).
        self.exact_apps = exact_apps
        # Apps gratis: el consumo se registra con su coste, pero no se cobra.
        self.free_apps = free_apps
        self.costs = ProviderCostCalculator()
        self.policy = BillingPolicy()

    def decide_exact(
        self,
        *,
        usage: NormalizedUsage,
        price: PriceCard,
        wallet_balance_cents: int,
        carry_microeur: int,
        free: bool = False,
    ) -> tuple[BillingDecision, int]:
        """Un consumo cobrado a su coste x margen exacto, sin mínimo (Borea).

        Devuelve la decisión y lo que queda por debajo del céntimo para el
        siguiente consumo. Gratis: se apunta el coste y no se cobra nada.
        """
        event_cost = self.costs.calculate_microusd(usage, price)
        provider_quote = self.policy.quote(event_cost, self.usd_to_eur, self.margin_multiplier)
        if free:
            decision = BillingDecision(
                provider_cost_microusd=event_cost,
                provider_cost_eur_cents=provider_quote.provider_cost_eur_cents,
                requested_charge_cents=0, charged_amount_cents=0,
                gross_margin_cents=-provider_quote.provider_cost_eur_cents,
                call_total_provider_cost_microusd=event_cost, call_total_charge_cents=0,
                partial_charge=False,
            )
            return decision, carry_microeur
        provider_microeur = Decimal(event_cost) * self.usd_to_eur
        if price.tool_call_charge_eur_cents is not None:
            tools_cost = Decimal(self.costs.tool_calls_microusd(usage.tool_calls, price))
            rest = (Decimal(event_cost) - tools_cost) * self.usd_to_eur * self.margin_multiplier
            fixed = Decimal(usage.tool_calls) * price.tool_call_charge_eur_cents * 10_000
            charge_microeur = max(rest, Decimal(0)) + fixed
        else:
            charge_microeur = provider_microeur * self.margin_multiplier
        total = carry_microeur + int(charge_microeur.quantize(Decimal("1"), rounding=ROUND_CEILING))
        requested = total // 10_000
        charged = min(requested, max(wallet_balance_cents, 0))
        # Lo que no se ha podido cobrar por falta de saldo no se arrastra: igual que
        # en Locus, el consumo se queda a medias y la app pide recargar.
        new_carry = total - requested * 10_000
        provider_cents = int(
            (provider_microeur / 10_000).quantize(Decimal("1"), rounding=ROUND_CEILING)
        )
        decision = BillingDecision(
            provider_cost_microusd=event_cost,
            provider_cost_eur_cents=provider_cents,
            requested_charge_cents=requested,
            charged_amount_cents=charged,
            gross_margin_cents=charged - provider_cents,
            call_total_provider_cost_microusd=event_cost,
            call_total_charge_cents=requested,
            partial_charge=charged < requested,
        )
        return decision, new_carry

    def decide(
        self,
        *,
        usage: NormalizedUsage,
        price: PriceCard,
        wallet_balance_cents: int,
        previous_provider_cost_microusd: int = 0,
        previous_charged_cents: int = 0,
        previous_tool_calls: int = 0,
        aggregate_by_call: bool = False,
    ) -> BillingDecision:
        event_cost = self.costs.calculate_microusd(usage, price)
        previous_cost = previous_provider_cost_microusd if aggregate_by_call else 0
        previous_charge = previous_charged_cents if aggregate_by_call else 0
        previous_calls = previous_tool_calls if aggregate_by_call else 0
        total_cost = previous_cost + event_cost

        total_quote = self.policy.quote(total_cost, self.usd_to_eur, self.margin_multiplier)
        previous_quote = self.policy.quote(previous_cost, self.usd_to_eur, self.margin_multiplier)
        total_charge = total_quote.charged_amount_cents
        if price.tool_call_charge_eur_cents is not None:
            # Las herramientas con precio fijo salen del margen: el resto de la
            # llamada se cobra a coste x margen y cada llamada a herramienta a
            # su precio, sobre el total de la llamada como todo lo demas.
            total_calls = previous_calls + usage.tool_calls
            rest_cost = total_cost - self.costs.tool_calls_microusd(total_calls, price)
            rest_quote = self.policy.quote(
                max(rest_cost, 0), self.usd_to_eur, self.margin_multiplier
            )
            fixed = (Decimal(total_calls) * price.tool_call_charge_eur_cents).quantize(
                Decimal("1"), rounding=ROUND_CEILING
            )
            total_charge = rest_quote.charged_amount_cents + int(fixed)
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
