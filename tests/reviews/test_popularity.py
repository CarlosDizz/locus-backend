"""El tamaño de los marcadores del mapa, a partir de visitas y «me gusta».

Lo que se fija: que el escalón es relativo a la ciudad y no absoluto; que un
sitio sin visitas no se castiga (sale normal, no pequeño); que hacen falta
unas cuantas visitas para destacar, para que una sola llamada no ponga un sitio
el primero; y que un sitio solo se encoge si a la mayoría no le gustó, con
votos suficientes para que no sea el mal día de una persona.
"""

import pytest
from pydantic import ValidationError

from locus_v2.reviews.service import PoiStats, ReviewInput, popularity_tiers


def _ciudad(**sitios: PoiStats) -> dict[int, PoiStats]:
    return {i: stat for i, stat in enumerate(sitios.values())}


def test_without_any_visits_everything_is_normal() -> None:
    stats = {i: PoiStats(visits=0, likes=0, votes=0) for i in range(20)}
    assert set(popularity_tiers(stats).values()) == {1}


def test_the_most_visited_and_liked_stand_out() -> None:
    stats = {i: PoiStats(visits=0, likes=0, votes=0) for i in range(20)}
    stats[0] = PoiStats(visits=40, likes=18, votes=19)  # el Coliseo
    stats[1] = PoiStats(visits=25, likes=10, votes=11)
    stats[2] = PoiStats(visits=12, likes=5, votes=6)
    stats[3] = PoiStats(visits=6, likes=3, votes=3)
    stats[4] = PoiStats(visits=4, likes=0, votes=0)

    tiers = popularity_tiers(stats)

    assert tiers[0] == 3 and tiers[1] == 3  # el 10 % de 20 son 2
    assert tiers[2] == 2 and tiers[3] == 2 and tiers[4] == 2
    assert tiers[10] == 1


def test_one_visit_is_not_enough_to_be_the_top() -> None:
    stats = {i: PoiStats(visits=0, likes=0, votes=0) for i in range(10)}
    stats[0] = PoiStats(visits=1, likes=1, votes=1)
    assert popularity_tiers(stats)[0] == 1


def test_a_place_most_people_disliked_shrinks() -> None:
    stats = {i: PoiStats(visits=0, likes=0, votes=0) for i in range(10)}
    stats[0] = PoiStats(visits=9, likes=1, votes=5)
    assert popularity_tiers(stats)[0] == 0


def test_one_bad_vote_does_not_sink_a_place() -> None:
    stats = {i: PoiStats(visits=0, likes=0, votes=0) for i in range(10)}
    stats[0] = PoiStats(visits=2, likes=0, votes=1)
    assert popularity_tiers(stats)[0] != 0


def test_empty_city() -> None:
    assert popularity_tiers({}) == {}


def test_the_review_trims_blank_comments_and_dedupes_tags() -> None:
    review = ReviewInput(
        place_vote=True, place_comment="   ", guide_tags=["learned", "learned", "too_long"]
    )
    assert review.place_comment is None
    assert review.guide_tags == ["learned", "too_long"]


@pytest.mark.parametrize(
    "datos",
    [{"guide_stars": 0}, {"guide_stars": 6}, {"guide_tags": ["inventada"]}],
)
def test_the_review_rejects_nonsense(datos: dict) -> None:
    with pytest.raises(ValidationError):
        ReviewInput(**datos)
