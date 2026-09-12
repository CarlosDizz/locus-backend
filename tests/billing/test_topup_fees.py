"""Lo que se queda la pasarela, que es lo que separa cobrar de ingresar.

El panel de recargas enseña un neto, y un neto que no se parezca al extracto es
peor que no enseñarlo. Estas pruebas fijan la única parte que podemos garantizar:
que la cuenta es la que decimos que es, y que la parte fija se cobra por cobro y
no por euro.
"""

from locus_v2.billing.topup_catalogue import (
    PADDLE_FEE_FIXED_CENTS,
    estimated_fee_cents,
    estimated_fees_for,
)


def test_paddle_charges_a_slice_plus_a_fixed_amount() -> None:
    assert estimated_fee_cents("paddle", 999) == round(999 * 0.05) + PADDLE_FEE_FIXED_CENTS


def test_google_play_is_a_clean_percentage() -> None:
    assert estimated_fee_cents("google_play", 1000) == 150


def test_a_provider_we_do_not_price_counts_as_free() -> None:
    # Mejor un bruto honesto que un neto inventado.
    assert estimated_fee_cents("manual", 1000) == 0
    assert estimated_fee_cents("una_pasarela_futura", 1000) == 0


def test_the_fixed_part_is_charged_once_per_payment() -> None:
    """Veinte recargas de 5 EUR no rinden como una de 100.

    Es la diferencia que justifica que el tramo más barato sea 4,99 y no 1 EUR,
    así que el panel tiene que contarla: sin el número de cobros, la parte fija
    desaparece y todo parece igual de rentable.
    """
    una = estimated_fees_for("paddle", 10000, 1)
    veinte = estimated_fees_for("paddle", 10000, 20)
    assert veinte - una == PADDLE_FEE_FIXED_CENTS * 19


def test_the_aggregate_matches_adding_up_one_by_one() -> None:
    importes = [499, 999, 1999]
    fila_a_fila = sum(estimated_fee_cents("paddle", importe) for importe in importes)
    agregado = estimated_fees_for("paddle", sum(importes), len(importes))
    # Redondear una suma no es lo mismo que sumar redondeos, pero la diferencia
    # no puede pasar de unos céntimos o el neto del panel dejaría de ser creíble.
    assert abs(fila_a_fila - agregado) <= 2


def test_nothing_charged_means_nothing_owed() -> None:
    assert estimated_fee_cents("paddle", 0) == 0
    assert estimated_fees_for("paddle", 0, 0) == 0
