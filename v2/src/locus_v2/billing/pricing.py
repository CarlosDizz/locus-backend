from decimal import Decimal, ROUND_CEILING

from pydantic import BaseModel, Field


class NormalizedUsage(BaseModel):
    text_input_tokens: int = Field(default=0, ge=0)
    cached_text_input_tokens: int = Field(default=0, ge=0)
    text_output_tokens: int = Field(default=0, ge=0)
    audio_input_tokens: int = Field(default=0, ge=0)
    audio_output_tokens: int = Field(default=0, ge=0)
    audio_input_milliseconds: int = Field(default=0, ge=0)
    audio_output_milliseconds: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    raw: dict = Field(default_factory=dict)


class PriceCard(BaseModel):
    text_input_per_million_usd: Decimal = Decimal(0)
    cached_text_input_per_million_usd: Decimal = Decimal(0)
    text_output_per_million_usd: Decimal = Decimal(0)
    audio_input_per_million_tokens_usd: Decimal = Decimal(0)
    audio_output_per_million_tokens_usd: Decimal = Decimal(0)
    audio_input_per_minute_usd: Decimal = Decimal(0)
    audio_output_per_minute_usd: Decimal = Decimal(0)
    tool_call_usd: Decimal = Decimal(0)


class ProviderCostCalculator:
    ONE_MILLION = Decimal(1_000_000)
    ONE_MINUTE_MS = Decimal(60_000)
    MICRO_USD = Decimal(1_000_000)

    def calculate_microusd(self, usage: NormalizedUsage, price: PriceCard) -> int:
        total = Decimal(0)
        total += Decimal(usage.text_input_tokens) * price.text_input_per_million_usd / self.ONE_MILLION
        total += (
            Decimal(usage.cached_text_input_tokens)
            * price.cached_text_input_per_million_usd
            / self.ONE_MILLION
        )
        total += Decimal(usage.text_output_tokens) * price.text_output_per_million_usd / self.ONE_MILLION
        total += (
            Decimal(usage.audio_input_tokens)
            * price.audio_input_per_million_tokens_usd
            / self.ONE_MILLION
        )
        total += (
            Decimal(usage.audio_output_tokens)
            * price.audio_output_per_million_tokens_usd
            / self.ONE_MILLION
        )
        total += (
            Decimal(usage.audio_input_milliseconds)
            * price.audio_input_per_minute_usd
            / self.ONE_MINUTE_MS
        )
        total += (
            Decimal(usage.audio_output_milliseconds)
            * price.audio_output_per_minute_usd
            / self.ONE_MINUTE_MS
        )
        total += Decimal(usage.tool_calls) * price.tool_call_usd
        return int((total * self.MICRO_USD).quantize(Decimal("1"), rounding=ROUND_CEILING))


class CustomerCharge(BaseModel):
    provider_cost_eur_cents: int
    charged_amount_cents: int
    gross_margin_cents: int


class BillingPolicy:
    def quote(
        self,
        provider_cost_microusd: int,
        usd_to_eur: Decimal,
        margin_multiplier: Decimal,
    ) -> CustomerCharge:
        provider_eur = Decimal(provider_cost_microusd) / Decimal(1_000_000) * usd_to_eur
        provider_cents = int((provider_eur * 100).quantize(Decimal("1"), rounding=ROUND_CEILING))
        charged_cents = int(
            (provider_eur * margin_multiplier * 100).quantize(Decimal("1"), rounding=ROUND_CEILING)
        )
        return CustomerCharge(
            provider_cost_eur_cents=provider_cents,
            charged_amount_cents=charged_cents,
            gross_margin_cents=charged_cents - provider_cents,
        )
