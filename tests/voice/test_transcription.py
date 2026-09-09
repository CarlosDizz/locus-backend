"""The WAV wrapper has to be right or the transcript comes back subtly wrong.

Getting the sample rate wrong does not raise: it produces audio that plays fast
or slow, and a transcription that is plausible and incorrect. That is the worst
kind of bug to have in a log nobody double-checks, so the header is asserted
rather than assumed.
"""

import io
import wave

import pytest

from locus_v2.voice.transcription import (
    SAMPLE_RATE_HZ,
    audio_seconds,
    pcm16_to_wav,
    transcribe_turn,
)


def test_wav_header_matches_what_we_actually_send() -> None:
    one_second = b"\x00\x01" * SAMPLE_RATE_HZ

    with wave.open(io.BytesIO(pcm16_to_wav(one_second)), "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE_HZ
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == SAMPLE_RATE_HZ


def test_audio_seconds_counts_real_time() -> None:
    assert audio_seconds(b"\x00\x01" * SAMPLE_RATE_HZ) == 1.0
    assert audio_seconds(b"") == 0.0


class FakeTranscriptions:
    def __init__(self) -> None:
        self.options: dict = {}

    async def create(self, **options):
        self.options = options
        return type("Response", (), {"text": "  hola que tal  "})()


class FakeClient:
    def __init__(self) -> None:
        self.audio = type("Audio", (), {"transcriptions": FakeTranscriptions()})()


@pytest.mark.asyncio
async def test_locale_is_reduced_to_the_language_code() -> None:
    client = FakeClient()

    result = await transcribe_turn(
        client, model="gpt-4o-mini-transcribe", pcm=b"\x00\x01" * 100, language="es-ES"
    )

    # La API espera ISO-639-1: mandarle "es-ES" la hace rechazar la peticion.
    assert client.audio.transcriptions.options["language"] == "es"
    assert result.text == "hola que tal"


@pytest.mark.asyncio
async def test_no_language_is_sent_when_there_is_none() -> None:
    client = FakeClient()

    await transcribe_turn(client, model="gpt-4o-mini-transcribe", pcm=b"\x00\x01" * 100)

    assert "language" not in client.audio.transcriptions.options
