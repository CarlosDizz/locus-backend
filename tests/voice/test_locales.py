"""A bare "es" made the guide sound Latin American to users in Spain.

Nothing failed and nothing was logged: the provider simply picked a generic
Spanish. The only way to catch this class of bug is to assert the exact tag we
ask for.
"""

from locus_v2.voice.locales import speech_locale


def test_bare_spanish_becomes_peninsular() -> None:
    assert speech_locale("es") == "es-ES"


def test_an_explicit_region_is_respected() -> None:
    # Un cliente que pide español de América sabe lo que quiere.
    assert speech_locale("es-419") == "es-419"
    assert speech_locale("pt-BR") == "pt-BR"


def test_underscores_and_case_are_normalised() -> None:
    assert speech_locale("es_mx") == "es-MX"
    assert speech_locale("EN-gb") == "en-GB"


def test_every_language_the_app_offers_has_a_region() -> None:
    for code in ("es", "en", "fr", "it", "de", "pt", "zh", "ja", "ar"):
        assert "-" in speech_locale(code), code


def test_unknown_and_empty_fall_back_to_spanish() -> None:
    assert speech_locale("") == "es-ES"
    assert speech_locale(None) == "es-ES"
    assert speech_locale("xx") == "es-ES"
