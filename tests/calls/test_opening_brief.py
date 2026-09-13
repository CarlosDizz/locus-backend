"""La ficha que el guía recibe sin haberla pedido.

Gemini documenta los lugares de memoria, y de memoria «Calle Feria» es la de
Sevilla y no la de Albacete. Su búsqueda nativa lo arreglaría, pero hoy tumba la
sesión entera con un 1007, así que documentamos aparte: una llamada a gpt-5-mini
en paralelo cuyo resultado se inyecta cuando está listo.

Lo que estas pruebas fijan es lo que no puede fallar nunca — que la llamada no
dependa de esto. Ni espera, ni se rompe si la ficha no llega, ni vuelve a pagar
por un sitio ya documentado.
"""

from datetime import timedelta

import pytest

from locus_v2.calls.bridge import BRIEF_METADATA_KEY, _cached_brief, _should_research
from locus_v2.shared.clock import utc_now


class _Poi:
    """Lo único que mira _cached_brief."""

    def __init__(self, metadata: dict | None) -> None:
        self.metadata_json = metadata


def _guardada(texto: str, *, hace_dias: float = 0) -> dict:
    return {
        BRIEF_METADATA_KEY: {
            "text": texto,
            "model": "gpt-5-mini",
            "created_at": (utc_now() - timedelta(days=hace_dias)).isoformat(),
        }
    }


def test_a_fresh_brief_is_reused() -> None:
    # El motivo de guardarla: la segunda llamada al mismo sitio no vuelve a pagar.
    poi = _Poi(_guardada("La plaza se abrió en 1783.", hace_dias=3))
    assert _cached_brief(poi, 90) == "La plaza se abrió en 1783."


def test_an_expired_brief_is_rewritten() -> None:
    poi = _Poi(_guardada("Texto viejo", hace_dias=120))
    assert _cached_brief(poi, 90) == ""


def test_a_place_with_no_brief_yet() -> None:
    assert _cached_brief(_Poi(None), 90) == ""
    assert _cached_brief(_Poi({}), 90) == ""
    assert _cached_brief(None, 90) == ""


@pytest.mark.parametrize("basura", [{"text": "   "}, {"text": ""}, "no soy un dict", 42, None])
def test_a_malformed_entry_is_treated_as_missing(basura: object) -> None:
    """Se rehace en vez de romper la llamada.

    Es metadata_json, un JSON libre que han tocado migraciones y el panel: no
    hay garantía de forma, y una llamada no puede caerse porque alguien dejara
    ahí algo raro.
    """
    assert _cached_brief(_Poi({BRIEF_METADATA_KEY: basura}), 90) == ""


def test_a_brief_with_no_usable_date_is_still_used() -> None:
    """Vieja pero útil vale más que ninguna.

    Sin fecha legible no se puede saber si caducó. Tirarla obligaría a pagar
    otra vez por algo que probablemente sigue siendo correcto.
    """
    poi = _Poi({BRIEF_METADATA_KEY: {"text": "Sirve igual", "created_at": "ayer por la tarde"}})
    assert _cached_brief(poi, 90) == "Sirve igual"


class _PoiConWikidata:
    def __init__(self, wikidata_id: str | None) -> None:
        self.wikidata_id = wikidata_id
        self.metadata_json: dict = {}


def test_a_place_wikidata_knows_is_not_researched() -> None:
    """El filtro que decide el gasto.

    El Pasaje de Lodares tiene entidad propia (Q5948330) y el modelo lo cuenta
    bien de memoria. Pagar por documentarlo sería pagar por lo que ya sale bien.
    """
    assert _should_research(_PoiConWikidata("Q5948330"), "", enabled=True) is False


def test_a_place_nobody_catalogued_is_researched() -> None:
    # Comprobado contra la API real: «Calle Feria (Albacete)» no existe en
    # Wikidata. Es justo la clase de sitio que acababa contando Sevilla.
    assert _should_research(_PoiConWikidata(""), "", enabled=True) is True
    assert _should_research(_PoiConWikidata(None), "", enabled=True) is True
    assert _should_research(_PoiConWikidata("   "), "", enabled=True) is True


def test_nothing_is_researched_twice() -> None:
    assert _should_research(_PoiConWikidata(""), "ya documentado", enabled=True) is False


def test_the_switch_turns_it_all_off() -> None:
    """Es un puente hasta que Google arregle su búsqueda, y se quita de un tirón."""
    assert _should_research(_PoiConWikidata(""), "", enabled=False) is False


def test_a_call_without_a_catalogued_place_researches_nothing() -> None:
    assert _should_research(None, "", enabled=True) is False
