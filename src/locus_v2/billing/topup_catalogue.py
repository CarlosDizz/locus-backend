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
