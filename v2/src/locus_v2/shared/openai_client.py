"""One place that builds OpenAI clients, so base URL and timeout apply everywhere.

V1 passed OPENAI_BASE_URL and OPENAI_RESPONSE_TIMEOUT_SECONDS to every call it
made. V2 had grown five separate `AsyncOpenAI(api_key=...)` call sites, none of
which honoured either, so a stuck request waited on the SDK default instead of
the 180s V1 would have allowed.

Two call sites deliberately do not use this:

- `voice/providers/openai_realtime.py` opens a *live session*, not a request. A
  voice call runs for minutes on purpose; putting a request timeout on it would
  cut the conversation off mid-sentence.
- `catalog/bootstrap/ai_client.py` is handed a bare api_key by callers that have
  no Settings, and it backs an admin-triggered background import where waiting
  longer is harmless. Threading Settings through those signatures buys nothing.
"""

from openai import AsyncOpenAI

from locus_v2.config import Settings


def build_openai_client(settings: Settings, api_key: str | None = None) -> AsyncOpenAI:
    key = api_key
    if key is None:
        if settings.openai_api_key is None:
            raise RuntimeError("The OpenAI API key is not configured")
        key = settings.openai_api_key.get_secret_value()
    return AsyncOpenAI(
        api_key=key,
        base_url=settings.openai_base_url or None,
        timeout=settings.openai_timeout_seconds,
    )
