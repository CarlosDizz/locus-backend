"""The transcription session payload is the contract with OpenAI.

Getting it wrong does not fail loudly: the session connects and simply
transcribes worse, which is the failure we spent an evening chasing. The point
of the second socket is `keywords` — the POI names — so that is what is
asserted.
"""

from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import LiveSessionConfig
from locus_v2.voice.providers.openai_transcribe import (
    MAX_KEYWORDS,
    _transcription_session,
)
from locus_v2.voice.transcription import build_vocabulary_prompt


def config(**overrides) -> LiveSessionConfig:
    base = {
        "model": "gpt-live-transcribe",
        "prompt": "Visita guiada en Pasaje de Lodares.",
        "locale": "es-ES",
        "audio_format": AudioFormat.PCM16_24KHZ,
        "tools": [],
        "provider_options": {"keywords": ["Pasaje de Lodares", "Museo de Albacete"]},
    }
    base.update(overrides)
    return LiveSessionConfig(**base)


def test_the_names_reach_the_session() -> None:
    session = _transcription_session(config())
    transcription = session["audio"]["input"]["transcription"]

    assert transcription["model"] == "gpt-live-transcribe"
    assert transcription["keywords"] == ["Pasaje de Lodares", "Museo de Albacete"]
    assert transcription["prompt"] == "Visita guiada en Pasaje de Lodares."


def test_language_is_the_bare_code() -> None:
    # La API espera ISO-639-1: "es-ES" no vale.
    transcription = _transcription_session(config())["audio"]["input"]["transcription"]
    assert transcription["languages"] == ["es"]


def test_the_room_owns_turn_taking_not_the_provider() -> None:
    # Si el proveedor tambien decidiera cuando acaba el turno, cortaria a la
    # gente a media frase: la sala ya sabe quien tiene la palabra.
    assert _transcription_session(config())["audio"]["input"]["turn_detection"] is None


def test_audio_format_matches_what_the_bridge_forwards() -> None:
    audio_format = _transcription_session(config())["audio"]["input"]["format"]
    assert audio_format == {"type": "audio/pcm", "rate": 24000}


def test_a_huge_vocabulary_is_trimmed() -> None:
    muchos = [f"Sitio {index}" for index in range(MAX_KEYWORDS + 25)]
    session = _transcription_session(config(provider_options={"keywords": muchos}))

    assert len(session["audio"]["input"]["transcription"]["keywords"]) == MAX_KEYWORDS


def test_no_keywords_means_the_field_is_absent() -> None:
    session = _transcription_session(config(provider_options={}))

    assert "keywords" not in session["audio"]["input"]["transcription"]


def test_vocabulary_prompt_names_the_place_and_drops_repeats() -> None:
    prompt = build_vocabulary_prompt(
        "Pasaje de Lodares", "Albacete", ["Museo de Albacete", "Pasaje de Lodares"]
    )

    assert prompt.count("Pasaje de Lodares") == 2  # el encabezado y la lista
    assert "Museo de Albacete" in prompt
    assert build_vocabulary_prompt("", "", []) == ""
