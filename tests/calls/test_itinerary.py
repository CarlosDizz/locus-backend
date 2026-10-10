"""Llamada en itinerario: qué sitio pasa, a qué lado, y cuándo contarlo.

Con coordenadas reales de Roma: un bus que baja por la Via dei Fori Imperiali
hacia el Coliseo. Lo que se fija: el lado sale del sentido de la marcha (no de
la brújula), un sitio que queda detrás no se cuenta, nunca se cuenta dos veces
el mismo, y entre dos sitios hay una pausa para no encadenarlos.
"""

from locus_v2.calls.itinerary import (
    Fix,
    ItineraryTracker,
    Place,
    bearing_deg,
    distance_m,
    narration_prompt,
    side_of,
)

COLISEO = Place("coliseo", "Coliseo", 41.8902, 12.4922, tier=3)
ARCO = Place("arco", "Arco de Constantino", 41.8898, 12.4906)
CIRCO = Place("circo", "Circo Máximo", 41.8861, 12.4851)
TREVI = Place("trevi", "Fontana di Trevi", 41.9009, 12.4833)


def _bus() -> ItineraryTracker:
    return ItineraryTracker(mode="bus", places=[COLISEO, ARCO, CIRCO, TREVI])


def test_distance_and_bearing_are_sane() -> None:
    assert 130 < distance_m(41.8902, 12.4922, 41.8898, 12.4906) < 150
    assert 85 < bearing_deg(41.89, 12.49, 41.89, 12.50) < 95  # hacia el este


def test_the_side_is_relative_to_where_you_are_going() -> None:
    # Hacia el norte, algo al este queda a la derecha; hacia el sur, a la izquierda.
    assert side_of(0, 90) == "right"
    assert side_of(180, 90) == "left"
    assert side_of(0, 10) == "ahead"
    assert side_of(0, 180) == "behind"
    assert side_of(None, 90) == "near"


def test_approaching_the_colosseum_by_bus() -> None:
    tracker = _bus()
    # Por la Via dei Fori Imperiali hacia el sureste, a unos 150 m del Coliseo.
    tracker.update(Fix(41.8912, 12.4908, t=0, heading=125, speed=5))

    ahora = tracker.now()
    assert ahora is not None and ahora.place.id == "coliseo"
    assert ahora.side in ("ahead", "left")

    contar = tracker.due(t=0)
    assert contar is not None and contar.place.id == "coliseo"
    assert tracker.due(t=1) is None  # pausa entre sitios
    despues = tracker.due(t=60)
    assert despues is None or despues.place.id != "coliseo"  # nunca dos veces


def test_what_is_behind_is_not_told() -> None:
    tracker = _bus()
    # Ya pasado el Coliseo, alejándose hacia el noroeste.
    tracker.update(Fix(41.8912, 12.4908, t=0, heading=305, speed=5))
    ahora = tracker.now()
    assert ahora is None or ahora.place.id != "coliseo"


def test_upcoming_is_ordered_by_arrival() -> None:
    tracker = _bus()
    # Saliendo del Coliseo hacia el suroeste, camino del Circo Máximo.
    tracker.update(Fix(41.8899, 12.4925, t=0, heading=230, speed=5))
    proximos = [s.place.id for s in tracker.upcoming()]
    assert "circo" in proximos
    assert "trevi" not in proximos  # queda detrás y lejos


def test_heading_is_worked_out_when_the_gps_gives_none() -> None:
    tracker = _bus()
    tracker.update(Fix(41.8930, 12.4870, t=0))
    tracker.update(Fix(41.8925, 12.4880, t=10))
    assert tracker.heading is not None and 100 < tracker.heading < 150


def test_the_prompt_gives_side_and_time_window() -> None:
    tracker = _bus()
    tracker.update(Fix(41.8912, 12.4908, t=0, heading=125, speed=5))
    texto = narration_prompt(tracker.due(t=0), "bus")
    assert "Coliseo" in texto and "segundos" in texto and "no la leas" in texto
