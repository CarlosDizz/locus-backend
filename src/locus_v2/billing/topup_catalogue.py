"""Qué se puede recargar desde la web, y por cuánto.

Vive aparte de cualquier pasarela porque no es asunto suyo: el importe lo decide
el servidor, y la pasarela solo cobra lo que se le diga. Cambiar de proveedor no
debería poder cambiar los precios por accidente.

Los tramos terminan en ,99 a propósito — es lo que espera ver quien compra — y
el suelo del importe libre no es redondo: sale de una cuenta.
"""


class TopUpError(RuntimeError):
    pass


# Los importes con botón propio.
WEB_TOPUP_PRODUCTS: dict[str, int] = {
    "web_top_up_5": 499,
    "web_top_up_10": 999,
    "web_top_up_20": 1999,
}

# El suelo del importe libre. Con Paddle (5% + 0,50 $ todo incluido) la comisión
# iguala al 15% de Google Play justo en 4,60 EUR: por debajo, cobrar directamente
# sale más caro que cobrar por la tienda, que es lo contrario de la intención.
# Por eso el tramo más bajo es 4,99 y no 2 EUR, como se planteó con Stripe —
# aquella cuenta salía con la comisión de Stripe, que es otra.
WEB_TOPUP_MIN_CENTS = 500
# Techo de partida, no una verdad revelada. 50 EUR son varias horas de llamada.
# Acota el fraude con tarjeta robada, que tantea con importes grandes, y limita
# la exposición a devoluciones: un saldo prepagado enorme es un pasivo.
WEB_TOPUP_MAX_CENTS = 5000

CURRENCY = "eur"


def resolve_amount_cents(product_id: str, amount_cents: int | None) -> int:
    """Decide qué se cobra. El único sitio donde esa respuesta puede salir.

    Un producto fijo se busca aquí; un importe libre se valida aquí. En los dos
    casos el número que llega a la pasarela es éste, nunca el que mandó el
    navegador, así que manipular el campo solo consigue pagar de verdad esa
    cantidad.
    """
    if product_id:
        amount = WEB_TOPUP_PRODUCTS.get(product_id)
        if amount is None:
            raise TopUpError("Producto de recarga no reconocido")
        return amount

    if amount_cents is None:
        raise TopUpError("Falta el importe de la recarga")
    if amount_cents % 100 != 0:
        raise TopUpError("El importe debe ser un número entero de euros")
    if amount_cents < WEB_TOPUP_MIN_CENTS:
        raise TopUpError(f"La recarga mínima es de {WEB_TOPUP_MIN_CENTS / 100:.2f} €")
    if amount_cents > WEB_TOPUP_MAX_CENTS:
        raise TopUpError(f"La recarga máxima es de {WEB_TOPUP_MAX_CENTS // 100} €")
    return amount_cents


# Lo que se queda la pasarela, para poder enseñar en el panel lo que de verdad
# llega a la cuenta y no solo lo que pagó el cliente. Son estimaciones nuestras,
# no el liquidado real: Paddle cobra 5% + 0,50 $ y el cambio se mueve, y Google
# se lleva un 15% limpio. Sirven para saber si un tramo compensa, no para cuadrar
# con el extracto.
PADDLE_FEE_RATE = 0.05
PADDLE_FEE_FIXED_CENTS = 46  # 0,50 $ a 0,87 EUR/USD, redondeado hacia arriba.
GOOGLE_PLAY_FEE_RATE = 0.15


def estimated_fee_cents(provider: str, amount_cents: int) -> int:
    """Comisión estimada de una recarga, en céntimos.

    Un proveedor que no conocemos se cuenta como comisión cero: inventar un
    número sería peor que enseñar el bruto y que se note que falta el dato.
    """
    if amount_cents <= 0:
        return 0
    if provider == "paddle":
        return round(amount_cents * PADDLE_FEE_RATE) + PADDLE_FEE_FIXED_CENTS
    if provider == "google_play":
        return round(amount_cents * GOOGLE_PLAY_FEE_RATE)
    return 0


def estimated_fees_for(provider: str, gross_cents: int, top_ups: int) -> int:
    """Lo mismo pero para un montón de recargas ya sumadas.

    La parte fija se cobra una vez por cobro, así que hace falta saber cuántos
    fueron: sin eso, un total de 100 EUR en veinte recargas parecería igual de
    rentable que en una sola, y no lo es ni de lejos.
    """
    if gross_cents <= 0 or top_ups <= 0:
        return 0
    if provider == "paddle":
        return round(gross_cents * PADDLE_FEE_RATE) + PADDLE_FEE_FIXED_CENTS * top_ups
    if provider == "google_play":
        return round(gross_cents * GOOGLE_PLAY_FEE_RATE)
    return 0
