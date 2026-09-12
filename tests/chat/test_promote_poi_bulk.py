"""Por qué el guía nunca subía nada al catálogo.

El chat tenía dos formas de enseñar un sitio: una marca temporal que se borra
al refrescar el mapa, y una promoción al catálogo con ficha y visita guiada.
En 72 horas de producción la primera se llamó tres veces y la segunda ninguna.

No estaba rota: costaba más. `promote_poi_to_catalog` aceptaba un solo nombre
por llamada, así que enseñar cinco museos eran cinco llamadas frente a una sola
de `mark_pois_on_map`. Con esa diferencia el modelo siempre elegía marcar, y el
usuario pedía POIs y recibía puntos que se desvanecían.

Estas pruebas fijan la parte que quita ese incentivo: una llamada, varios
sitios — sin romper el nombre suelto, que es lo que manda el modelo cuando solo
hay uno.
"""

import asyncio
from typing import Any

from locus_v2.chat.tools import ChatToolDispatcher


class _Recorder(ChatToolDispatcher):
    """Se queda con los nombres que llegan abajo, sin tocar base de datos.

    Hereda para probar el reparto real de `_promote_poi` — que es lo que
    cambió — y sustituye solo el escalón siguiente.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, bool]] = []
        self.refuse: set[str] = set()

    async def _promote_one(
        self, poi_name: str, arguments: dict[str, Any], *, set_active: bool
    ) -> dict[str, Any]:
        self.seen.append((poi_name, set_active))
        if poi_name in self.refuse:
            return {"ok": False, "poi_name": poi_name, "error": "not_a_landmark"}
        return {"ok": True, "poi_name": poi_name, "status": "promoted_to_catalog"}


def _run(arguments: dict[str, Any]) -> tuple[_Recorder, dict[str, Any]]:
    recorder = _Recorder()
    result = asyncio.run(recorder._promote_poi(arguments))
    return recorder, result


def test_several_places_go_up_in_a_single_call() -> None:
    recorder, result = _run({"poi_names": ["Museo del Prado", "Reina Sofía", "Thyssen"]})
    assert [name for name, _ in recorder.seen] == [
        "Museo del Prado",
        "Reina Sofía",
        "Thyssen",
    ]
    assert result["ok"] is True
    assert result["promoted_count"] == 3


def test_a_bare_name_still_works() -> None:
    # Es lo que manda el modelo cuando solo hay un sitio, y lo que mandaban
    # todas las llamadas anteriores a este cambio.
    recorder, result = _run({"poi_name": "Museo del Prado"})
    assert [name for name, _ in recorder.seen] == ["Museo del Prado"]
    assert result["promoted_count"] == 1


def test_only_a_lone_promotion_steals_the_focus() -> None:
    """Con cinco museos, dejar de foco al último es arbitrario.

    `set_active` mueve el sitio del que va la conversación, así que en lote se
    apaga: pisaría aquello de lo que se estaba hablando por el mero orden de la
    lista.
    """
    solo, _ = _run({"poi_names": ["Museo del Prado"]})
    assert solo.seen == [("Museo del Prado", True)]

    lote, _ = _run({"poi_names": ["Museo del Prado", "Thyssen"]})
    assert [activo for _, activo in lote.seen] == [False, False]


def test_one_bar_in_the_list_does_not_sink_the_rest() -> None:
    recorder = _Recorder()
    recorder.refuse = {"Bar Manolo"}
    result = asyncio.run(
        recorder._promote_poi({"poi_names": ["Catedral", "Bar Manolo", "Alcázar"]})
    )
    assert result["ok"] is True
    assert result["promoted_count"] == 2
    # El rechazo viaja de vuelta para que el guía pueda decir por qué ese no.
    assert [item["ok"] for item in result["results"]] == [True, False, True]


def test_asking_for_nothing_is_refused_before_any_work() -> None:
    for arguments in ({}, {"poi_names": []}, {"poi_name": "   "}):
        recorder, result = _run(arguments)
        assert result["ok"] is False
        assert recorder.seen == []
