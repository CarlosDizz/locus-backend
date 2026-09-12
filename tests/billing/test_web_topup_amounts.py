"""What a web top-up is allowed to charge.

The fixed buttons were always safe: the amount came from a server-side table and
the browser only named a product. The free-amount field gives that up, so the
guarantee has to move somewhere else — the server validates the proposal and is
the one that tells the gateway what to charge. These tests pin that boundary, because
getting it wrong is the difference between selling credit and giving it away.
"""

import pytest

from locus_v2.billing.topup_catalogue import (
    WEB_TOPUP_MAX_CENTS,
    WEB_TOPUP_MIN_CENTS,
    WEB_TOPUP_PRODUCTS,
    TopUpError,
    resolve_amount_cents,
)


def test_a_fixed_product_takes_its_price_from_the_server() -> None:
    # El navegador nombra un producto; el importe no viaja nunca desde el.
    assert resolve_amount_cents("web_top_up_5", None) == 499
    assert resolve_amount_cents("web_top_up_5", 999999) == 499


def test_an_unknown_product_is_refused() -> None:
    with pytest.raises(TopUpError):
        resolve_amount_cents("web_top_up_1000", None)


def test_no_tier_is_cheap_enough_to_lose_money_on() -> None:
    # Con Paddle (5% + 0,50 $) la comision iguala al 15% de Play en 4,60 EUR.
    # Por debajo de ahi, cobrar por web sale mas caro que cobrar por la tienda.
    assert min(WEB_TOPUP_PRODUCTS.values()) == 499
    for importe in WEB_TOPUP_PRODUCTS.values():
        comision = importe * 0.05 + 46
        assert comision / importe <= 0.15


def test_a_custom_amount_is_accepted_between_the_bounds() -> None:
    assert resolve_amount_cents("", WEB_TOPUP_MIN_CENTS) == WEB_TOPUP_MIN_CENTS
    assert resolve_amount_cents("", WEB_TOPUP_MAX_CENTS) == WEB_TOPUP_MAX_CENTS
    assert resolve_amount_cents("", 1200) == 1200


@pytest.mark.parametrize(
    "amount",
    [
        WEB_TOPUP_MIN_CENTS - 100,  # por debajo del minimo
        WEB_TOPUP_MAX_CENTS + 100,  # por encima del maximo
        -500,  # negativo
        250,  # no es un numero entero de euros
        0,
    ],
)
def test_a_custom_amount_outside_the_rules_never_reaches_the_gateway(amount: int) -> None:
    with pytest.raises(TopUpError):
        resolve_amount_cents("", amount)


def test_asking_for_nothing_at_all_is_refused() -> None:
    with pytest.raises(TopUpError):
        resolve_amount_cents("", None)


def test_the_minimum_is_where_paddle_stops_being_worse_than_the_store() -> None:
    # 0,50 $ fijos + 5%. Play se lleva el 15%. Por debajo de este punto, cobrar
    # por web cuesta mas que cobrar por la tienda.
    comision = 46 + WEB_TOPUP_MIN_CENTS * 0.05
    assert comision / WEB_TOPUP_MIN_CENTS <= 0.15


def test_the_cheapest_tier_is_below_the_custom_minimum() -> None:
    """El fallo que rechazaba todos los pagos de 4,99 €.

    El suelo del importe libre son 500 céntimos y el tramo más barato son 499.
    Mientras la comprobación de rango se aplicó también a los tramos del
    catálogo, cada pago de 4,99 € se cobraba en la pasarela y se rechazaba al
    abonar. Que el catálogo cruce ese suelo no es un error — es lo que hace
    falta para que los precios acaben en ,99 — pero obliga a validar cada cosa
    por su lado: un tramo contra su precio exacto, el importe libre contra el
    rango.
    """
    assert min(WEB_TOPUP_PRODUCTS.values()) < WEB_TOPUP_MIN_CENTS
