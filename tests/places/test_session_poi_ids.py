"""Los sitios que llegan a la sesión del mapa llevan el id público, no el interno.

En producción (2026-10-15) una llamada al Cupertino Historical Society & Museum
(id interno 879, creado desde el chat) arrancó hablando del Monumento a los
Coloraos de Almería, cuyo id heredado de V1 es también 879. La app manda ese id
a `POST /api/calls`, y allí se interpreta como id público: para los sitios con
id de V1 es ese id; para los nuevos, 1.000.000.000 + id interno.
"""

from locus_v2.catalog.models import Poi
from locus_v2.chat.tools import _catalog_row_to_session_poi
from locus_v2.places.service import _catalog_poi
from locus_v2.shared.mobile_ids import MOBILE_ID_OFFSET


def build_poi(poi_id: int, legacy_v1_id: int | None) -> Poi:
    return Poi(
        id=poi_id,
        legacy_v1_id=legacy_v1_id,
        name="Cupertino Historical Society & Museum",
        names_json={},
        short_descriptions_json={},
        lat=37.3257943,
        lng=-122.0424345,
        short_description="",
        long_description="",
        google_place_id="",
    )


def test_new_poi_gets_the_offset_id_in_both_paths() -> None:
    poi = build_poi(879, legacy_v1_id=None)

    assert _catalog_poi(poi, "es").id == str(MOBILE_ID_OFFSET + 879)
    assert _catalog_row_to_session_poi(poi).id == str(MOBILE_ID_OFFSET + 879)


def test_poi_from_v1_keeps_its_legacy_id() -> None:
    poi = build_poi(853, legacy_v1_id=879)

    assert _catalog_poi(poi, "es").id == "879"
    assert _catalog_row_to_session_poi(poi).id == "879"
