"""Que la visita sea del sitio correcto, y no del homónimo más famoso.

Una llamada sobre la Calle Feria de Albacete acabó contando la Calle Feria de
Sevilla, y otra sobre la Plaza Altozano se fue a la de Triana. El catálogo tenía
el dato bien —ciudad Albacete, coordenadas de Albacete— pero el prompt de la
llamada solo interpolaba `{poi_name}`: `{city_name}` y `{poi_description}` se
calculaban, se pasaban a render_prompt y la plantilla no los usaba. Con el
nombre a secas y la orden de contar de memoria, el modelo rellenaba el hueco con
lo que sabe del mundo, que es siempre el sitio más conocido.

El bloque vive en código y no en el prompt por lo mismo que el perfil del
viajero: de esto depende que la visita no sea de otra ciudad, y no debería poder
desaparecer con una edición descuidada desde el panel.
"""

from locus_v2.calls.bridge import _poi_location_block


def test_the_city_travels_with_the_name() -> None:
    bloque = _poi_location_block(
        {"name": "Calle Feria", "city_name": "Albacete", "description": "C/ Feria, Albacete"}
    )
    assert "Calle Feria" in bloque
    assert "Albacete" in bloque


def test_it_says_out_loud_that_the_famous_one_is_not_this_one() -> None:
    """Nombrar la ciudad no basta: el modelo ya la tenía en la ficha del catálogo.

    Lo que faltaba era la instrucción de qué hacer cuando su memoria contradice
    la ficha — que es exactamente el caso de un homónimo célebre.
    """
    bloque = _poi_location_block({"name": "Plaza Altozano", "city_name": "Albacete"})
    assert "otras ciudades" in bloque
    assert "no lo cuentes" in bloque


def test_a_place_with_no_data_gets_no_block() -> None:
    # Un "Ciudad:" vacío invita a rellenar el hueco, que es el fallo que esto evita.
    assert _poi_location_block({}) == ""
    assert _poi_location_block({"name": "", "city_name": "", "description": ""}) == ""
    assert _poi_location_block({"name": "  ", "city_name": None}) == ""


def test_what_we_do_know_survives_what_we_do_not() -> None:
    bloque = _poi_location_block({"name": "Pasaje de Lodares", "city_name": "", "description": ""})
    assert "Pasaje de Lodares" in bloque
    assert "Ciudad:" not in bloque
