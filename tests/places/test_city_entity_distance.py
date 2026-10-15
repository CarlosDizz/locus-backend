"""El import desde la entidad de Wikidata no mete sitios de otra ciudad homónima.

En producción (2026-10-15) «San José» (California) se resolvió por nombre a
San José de Costa Rica, y su catedral y su Teatro Nacional entraron en el
catálogo colgando de la ciudad californiana.
"""

from locus_v2.catalog.bootstrap.service import _near_city
from locus_v2.catalog.models import City


def row(lat: float, lng: float, name: str) -> dict[str, dict[str, str]]:
    return {
        "coord": {"value": f"Point({lng} {lat})"},
        "poiLabel": {"value": name},
    }


def test_rows_from_a_namesake_city_are_dropped() -> None:
    san_jose_ca = City(name="San José", lat=37.3382, lng=-121.8863)
    rows = [
        row(9.93275, -84.0787889, "Catedral Metropolitana de San José"),
        row(37.3333, -121.8907, "San Jose Museum of Art"),
    ]

    kept = _near_city(san_jose_ca, rows, radius_km=8)

    assert [item["poiLabel"]["value"] for item in kept] == ["San Jose Museum of Art"]


def test_city_without_coordinates_keeps_everything() -> None:
    city = City(name="San José", lat=None, lng=None)
    rows = [row(9.93275, -84.0787889, "Catedral Metropolitana de San José")]

    assert _near_city(city, rows, radius_km=8) == rows
