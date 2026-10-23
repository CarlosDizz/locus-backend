"""Notas de voz en el chat de Borea, como en BACUL y en WhatsApp.

El audio se transcribe con Gemini 3.1 Flash-Lite (GPT-6 Luna no escucha audio),
se descarta y lo que llega al planificador es el texto. La transcripción se
cobra aparte, como un consumo más de la misma app.
"""

import base64
from typing import Any

import httpx
import structlog

from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.config import Settings

logger = structlog.get_logger()

TRANSCRIBE_MODEL = "gemini-3.1-flash-lite"
_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_VOICE_BYTES = 8 * 1024 * 1024

_INSTRUCTIONS = (
    "Transcribe literalmente esta nota de voz de alguien que prepara un viaje. Solo el "
    "texto, en el idioma en que habla, con puntuación normal. Sin comentarios ni comillas. "
    "Si no se oye nada, responde vacío."
)


class VoiceError(Exception):
    pass


async def transcribe(settings: Settings, audio: bytes, mime: str) -> tuple[str, NormalizedUsage]:
    if settings.gemini_api_key is None:
        raise VoiceError("Las notas de voz no están disponibles")
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": _INSTRUCTIONS},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio).decode()}},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 1200},
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{_URL}/{TRANSCRIBE_MODEL}:generateContent",
                headers={"x-goog-api-key": settings.gemini_api_key.get_secret_value().strip()},
                json=body,
            )
    except httpx.HTTPError as error:
        logger.warning("borea_voice_failed", error=str(error))
        raise VoiceError("No he podido escuchar la nota de voz") from error
    if response.status_code != 200:
        logger.warning("borea_voice_failed", status=response.status_code, body=response.text[:300])
        raise VoiceError("No he podido escuchar la nota de voz")
    data: dict[str, Any] = response.json()
    candidate = (data.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = " ".join(str(p.get("text") or "") for p in parts if not p.get("thought")).strip()
    return text, _usage(data)


def _usage(data: dict[str, Any]) -> NormalizedUsage:
    meta = data.get("usageMetadata") or {}
    audio = sum(
        int(d.get("tokenCount") or 0)
        for d in meta.get("promptTokensDetails") or []
        if d.get("modality") == "AUDIO"
    )
    prompt = int(meta.get("promptTokenCount") or 0)
    return NormalizedUsage(
        text_input_tokens=max(prompt - audio, 0),
        audio_input_tokens=audio,
        text_output_tokens=int(meta.get("candidatesTokenCount") or 0)
        + int(meta.get("thoughtsTokenCount") or 0),
    )
