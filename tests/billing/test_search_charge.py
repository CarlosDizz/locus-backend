"""La búsqueda de Google en las llamadas: se cobra cada consulta a precio fijo.

Con Gemini 3, Google cobra cada consulta (no cada pregunta) a 14 $ por mil
pasadas 5.000 gratis al mes, y el modelo decide cuántas lanza. Al cliente se le
cobran 1,5 céntimos de euro por consulta desde la primera, fuera del margen x2,2
que lleva el resto de la llamada. Lo que se fija: el precio por consulta, que el
redondeo va sobre el total de la llamada (si no, cada consulta saldría a 2
céntimos), que el coste real queda a la vista, y que una consulta repetida en
el mismo turno se cuenta una vez.
"""

from decimal import Decimal

from google.genai import types

from locus_v2.billing.application.processor import BillingDecisionEngine
from locus_v2.billing.pricing import NormalizedUsage, PriceCard
from locus_v2.voice.providers.gemini_live import _search_usage

BUSQUEDA = PriceCard(
    audio_output_per_million_tokens_usd=Decimal("12"),
    tool_call_usd=Decimal("0.014"),
    tool_call_charge_eur_cents=Decimal("1.5"),
)


def engine() -> BillingDecisionEngine:
    return BillingDecisionEngine(
        usd_to_eur=Decimal("0.87"),
        margin_multiplier=Decimal("2.20"),
        minimum_realtime_call_charge_cents=0,
    )


def test_each_query_costs_one_and_a_half_cents() -> None:
    decision = engine().decide(
        usage=NormalizedUsage(tool_calls=2),
        price=BUSQUEDA,
        wallet_balance_cents=100,
        aggregate_by_call=True,
    )
    assert decision.charged_amount_cents == 3
    # Lo que cobra Google de verdad, para que el panel no mienta sobre el margen.
    assert decision.provider_cost_microusd == 28_000


def test_rounding_is_over_the_whole_call() -> None:
    cobrado = 0
    coste = 0
    consultas = 0
    for _ in range(4):
        decision = engine().decide(
            usage=NormalizedUsage(tool_calls=1),
            price=BUSQUEDA,
            wallet_balance_cents=100,
            previous_provider_cost_microusd=coste,
            previous_charged_cents=cobrado,
            previous_tool_calls=consultas,
            aggregate_by_call=True,
        )
        cobrado += decision.charged_amount_cents
        coste += decision.provider_cost_microusd
        consultas += 1
    assert cobrado == 6  # 4 x 1,5, no 4 x 2


def test_the_rest_of_the_call_keeps_its_margin() -> None:
    sin_busqueda = engine().decide(
        usage=NormalizedUsage(audio_output_tokens=100_000),
        price=BUSQUEDA,
        wallet_balance_cents=1000,
        aggregate_by_call=True,
    )
    con_busqueda = engine().decide(
        usage=NormalizedUsage(audio_output_tokens=100_000, tool_calls=2),
        price=BUSQUEDA,
        wallet_balance_cents=1000,
        aggregate_by_call=True,
    )
    assert con_busqueda.charged_amount_cents == sin_busqueda.charged_amount_cents + 3


def test_without_a_fixed_price_tools_go_through_the_margin() -> None:
    card = PriceCard(tool_call_usd=Decimal("0.01"))
    decision = engine().decide(
        usage=NormalizedUsage(tool_calls=1),
        price=card,
        wallet_balance_cents=100,
    )
    assert decision.charged_amount_cents == 2  # 0,01 $ x 0,87 x 2,2 = 1,9 -> 2


def test_the_price_card_reads_the_fixed_price() -> None:
    card = PriceCard.from_json({"tool_call_usd": "0.014", "tool_call_charge_eur_cents": "1.5"})
    assert card.tool_call_charge_eur_cents == Decimal("1.5")
    assert PriceCard.from_json({}).tool_call_charge_eur_cents is None


def _mensaje(consultas: list[str], fin: bool = False) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            grounding_metadata=types.GroundingMetadata(web_search_queries=consultas),
            turn_complete=fin,
        )
    )


def test_a_query_repeated_in_the_same_turn_is_counted_once() -> None:
    vistas: set[str] = set()
    primero = _search_usage(_mensaje(["tiempo Albacete", "pronostico Albacete"]), vistas)
    repetido = _search_usage(_mensaje(["tiempo Albacete"], fin=True), vistas)
    otro_turno = _search_usage(_mensaje(["tiempo Albacete"]), vistas)

    assert primero is not None and primero.usage.tool_calls == 2
    assert repetido is None
    # Turno nuevo: si vuelve a buscarlo, Google vuelve a cobrarlo.
    assert otro_turno is not None and otro_turno.usage.tool_calls == 1


def test_a_message_without_searches_bills_nothing() -> None:
    assert _search_usage(types.LiveServerMessage(), set()) is None
