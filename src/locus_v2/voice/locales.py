"""Turn a bare language code into the regional one the speech provider needs.

Clients send "es". Gemini accepts it, so nothing fails — it just picks a
generic Spanish, which in practice comes out Latin American. Our users are in
Spain and it sounded wrong to them. The accent is chosen entirely by
`speech_config.language_code`, so the fix is to be explicit.

Anything already regional ("es-419", "pt-BR") is passed through untouched: a
client that knows what it wants is not second-guessed.
"""

from __future__ import annotations

# Región por defecto de cada idioma que ofrece la app. Son decisiones de
# producto, no técnicas: el público está en España y en Europa.
DEFAULT_REGIONS = {
    "es": "es-ES",
    "en": "en-GB",
    "fr": "fr-FR",
    "it": "it-IT",
    "de": "de-DE",
    "pt": "pt-PT",
    "zh": "zh-CN",
    "ja": "ja-JP",
    # Google usa una etiqueta pan-árabe en vez de un país concreto.
    "ar": "ar-XA",
}


def speech_locale(language: str | None) -> str:
    """Return the BCP-47 tag to ask the provider for.

    Falls back to Spanish for anything unrecognised, matching the rest of the
    app's default rather than leaving the provider to guess.
    """
    code = (language or "").strip()
    if not code:
        return DEFAULT_REGIONS["es"]
    if "-" in code or "_" in code:
        # Ya trae región: se respeta, solo se normaliza la forma.
        base, _, region = code.replace("_", "-").partition("-")
        return f"{base.lower()}-{region.upper()}"
    return DEFAULT_REGIONS.get(code.lower(), DEFAULT_REGIONS["es"])
