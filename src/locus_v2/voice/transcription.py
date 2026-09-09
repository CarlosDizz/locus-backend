"""Transcribe a finished voice turn with a dedicated speech-to-text model.

The live provider already hands us a transcript of what the user said, but its
quality is poor enough to be a problem: the shared log is what the group reads,
and `_recap_entries` replays those same lines to rebuild context whenever a
provider session reconnects. A bad transcript is not only ugly, it degrades the
conversation after every reconnection.

This runs *after* the turn, on the audio we already have in hand, so it adds no
moving parts to the live path: one HTTP request per turn, and if it fails the
call is unaffected.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

# Lo que manda la app y lo que se reenvía al proveedor (ver AudioFormat.PCM16_24KHZ
# en calls/bridge.py). Va explícito porque el WAV necesita saberlo para no sonar
# acelerado o lento, y un error aquí no da excepción: da una transcripción rara.
SAMPLE_RATE_HZ = 24000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1


@dataclass(slots=True)
class TurnTranscription:
    text: str
    audio_seconds: float


def pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Wrap raw PCM in a WAV container.

    The transcription endpoint takes files, not naked samples: it needs the
    header to know the rate and width. Written by hand rather than with a
    library because it is 44 bytes and stdlib already does it.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def audio_seconds(pcm: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> float:
    return len(pcm) / (sample_rate * SAMPLE_WIDTH_BYTES * CHANNELS)


async def transcribe_turn(
    client,
    *,
    model: str,
    pcm: bytes,
    language: str | None = None,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> TurnTranscription:
    """Return what was said in this turn. Raises whatever the client raises."""
    wav = pcm16_to_wav(pcm, sample_rate)
    # The SDK reads the name to pick a MIME type, so it has to end in .wav.
    payload = ("turn.wav", wav, "audio/wav")
    options: dict = {"model": model, "file": payload}
    if language:
        # Solo el código base: la API espera ISO-639-1 ("es"), no "es-ES".
        options["language"] = language.split("-")[0].lower()
    response = await client.audio.transcriptions.create(**options)
    return TurnTranscription(
        text=(getattr(response, "text", "") or "").strip(),
        audio_seconds=audio_seconds(pcm, sample_rate),
    )
