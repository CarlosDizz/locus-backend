from decimal import Decimal

from locus_v2.billing.application.processor import BillingDecisionEngine
from locus_v2.billing.pricing import NormalizedUsage, PriceCard


def engine() -> BillingDecisionEngine:
    return BillingDecisionEngine(
        usd_to_eur=Decimal("0.87"),
        margin_multiplier=Decimal("2.20"),
        minimum_realtime_call_charge_cents=3,
    )


def test_realtime_minimum_is_applied_once_per_call() -> None:
    price = PriceCard(audio_output_per_million_tokens_usd=Decimal("12"))
    first = engine().decide(
        usage=NormalizedUsage(audio_output_tokens=100),
        price=price,
        wallet_balance_cents=100,
        aggregate_by_call=True,
    )
    second = engine().decide(
        usage=NormalizedUsage(audio_output_tokens=100),
        price=price,
        wallet_balance_cents=97,
        previous_provider_cost_microusd=first.call_total_provider_cost_microusd,
        previous_charged_cents=first.charged_amount_cents,
        aggregate_by_call=True,
    )

    assert first.charged_amount_cents == 3
    assert second.charged_amount_cents == 0
    assert second.call_total_charge_cents == 3


def test_charge_is_capped_to_wallet_balance() -> None:
    decision = engine().decide(
        usage=NormalizedUsage(audio_output_tokens=1_000_000),
        price=PriceCard(audio_output_per_million_tokens_usd=Decimal("12")),
        wallet_balance_cents=4,
        aggregate_by_call=True,
    )

    assert decision.requested_charge_cents > 4
    assert decision.charged_amount_cents == 4
    assert decision.partial_charge is True


def test_legacy_price_keys_are_supported() -> None:
    card = PriceCard.from_json(
        {
            "text_input_per_million": "0.25",
            "text_cached_input_per_million": "0.025",
            "text_output_per_million": "2.00",
            "audio_input_per_million": "3.00",
            "audio_output_per_million": "12.00",
        }
    )

    assert card.text_input_per_million_usd == Decimal("0.25")
    assert card.audio_output_per_million_tokens_usd == Decimal("12.00")


def test_cached_audio_uses_its_own_rate() -> None:
    cost = engine().costs.calculate_microusd(
        NormalizedUsage(audio_input_tokens=100, cached_audio_input_tokens=900),
        PriceCard(
            audio_input_per_million_tokens_usd=Decimal("10"),
            cached_audio_input_per_million_tokens_usd=Decimal("0.30"),
        ),
    )

    assert cost == 1270
