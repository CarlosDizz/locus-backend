"""Qué se puede guardar en el catálogo y qué no.

El filtro buscaba subcadenas sobre trece palabras, y «bar» suelto aparece dentro
de Barcelona, Barrio, Barroco y Bárbara. Mientras el chat no promocionó nunca
nada, eso no se notó. En cuanto promocionar pasó a ser la vía normal para los
sitios visitables, este filtro empezó a decidir qué entra en el catálogo
compartido — y habría dejado fuera la catedral de una ciudad entera por el
nombre de la ciudad.

Los dos sentidos importan por igual y fallan distinto: un falso positivo deja
fuera un monumento y el usuario no entiende por qué; un falso negativo mete una
hamburguesería en el catálogo de todos, y eso no se deshace solo.
"""

import pytest

from locus_v2.places.service import looks_like_service

# Sitios visitables cuyo nombre contiene una etiqueta de servicio como trozo de
# otra palabra. Todos daban positivo antes.
VISITABLES = [
    "Catedral de Barcelona",
    "Barrio Gótico",
    "Museo Barroco",
    "Basílica de Santa Bárbara",
    "Barbacana de Ávila",
    "Puerta del Sol",
    "Mirador de San Nicolás",
    "Jardines de Sabatini",
]

SERVICIOS = [
    "Hotel Ritz",
    "Restaurante Botín",
    "Bar Manolo",
    "Cafetería Gijón",
    "Café Central",
    "Farmacia Central",
    "Pizzería Da Nico",
    "Taberna La Bola",
]


@pytest.mark.parametrize("nombre", VISITABLES)
def test_a_landmark_is_not_mistaken_for_a_bar(nombre: str) -> None:
    assert looks_like_service(nombre) is False


@pytest.mark.parametrize("nombre", SERVICIOS)
def test_a_service_is_still_caught(nombre: str) -> None:
    assert looks_like_service(nombre) is True


@pytest.mark.parametrize(
    "nombre",
    ["Los Bares de Huertas", "Hoteles Riu", "Restaurantes del Puerto"],
)
def test_the_plural_does_not_slip_through(nombre: str) -> None:
    """El límite de palabra deja fuera el plural si no se cuenta con él.

    «bar» entero no está dentro de «bares», así que sin el sufijo opcional el
    arreglo del falso positivo habría abierto un agujero por el otro lado.
    """
    assert looks_like_service(nombre) is True


def test_it_reads_the_whole_text_not_just_the_name() -> None:
    # El promotor le pasa nombre + descripcion + resumen juntos: un sitio con
    # nombre inocente y descripcion de restaurante tiene que caer igual.
    assert looks_like_service("Casa Lucio — restaurante castellano de 1974") is True
