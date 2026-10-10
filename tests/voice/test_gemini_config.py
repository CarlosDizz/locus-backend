"""The Gemini session config decides what a call remembers and whether tools block it.

There is deliberately no context window compression. With a 6000/3500 sliding
window and a system prompt past 3400 tokens, every trim left room for about 50
tokens of conversation: on 2026-10-06 the guide forgot the whole visit four
times in 25 minutes. Bringing the field back — even from a stale value saved in
a prompt's runtime config — would do that again, silently. Hence the test.
"""

import pytest
from google.genai import types

from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import LiveSessionConfig
from locus_v2.voice.providers.gemini_live import (
    GeminiLive3Provider,
    _gemini2_config,
    _gemini3_config,
)


def config(**overrides) -> LiveSessionConfig:
    base = {
        "model": "gemini-3.1-flash-live-preview",
        "prompt": "Eres un guia turistico.",
        "locale": "es-ES",
        "audio_format": AudioFormat.PCM16_24KHZ,
        "tools": [],
        "provider_options": {},
    }
    base.update(overrides)
    return LiveSessionConfig(**base)


def test_neither_family_compresses_the_context() -> None:
    stale = {"trigger_tokens": 6000, "sliding_window": {"target_tokens": 3500}}
    for build in (_gemini3_config, _gemini2_config):
        assert "context_window_compression" not in build(config())
        built = build(config(provider_options={"context_window_compression": stale}))
        assert "context_window_compression" not in built


def test_manual_turns_explicitly_interrupt_the_current_answer() -> None:
    built = _gemini3_config(
        config(provider_options={"turn_detection": {"type": "manual"}})
    )

    realtime = built["realtime_input_config"]
    assert realtime["automatic_activity_detection"] == {"disabled": True}
    assert (
        realtime["activity_handling"]
        == types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
    )


def _declarations(live_config: dict) -> list[dict]:
    return next(
        tool["function_declarations"]
        for tool in live_config["tools"]
        if "function_declarations" in tool
    )


def test_the_guide_can_search_with_or_without_our_tools() -> None:
    # Sin compresion, google_search ya no tumba la sesion (era la combinacion
    # de las dos lo que daba 1007). Se declara siempre: un guia sin forma de
    # comprobar un dato es justamente el que se lo inventa.
    tools = [{"name": "plan_poi_visit", "description": "Planea", "parameters": {}}]
    for build in (_gemini3_config, _gemini2_config):
        assert {"google_search": {}} in build(config())["tools"]
        built = build(config(tools=tools))
        assert {"google_search": {}} in built["tools"]
        assert _declarations(built)[0]["name"] == "plan_poi_visit"


def test_document_poi_is_non_blocking_only_on_gemini_38() -> None:
    tools = [
        {"name": "document_poi", "description": "Investiga", "parameters": {}},
        {"name": "plan_poi_visit", "description": "Planea", "parameters": {}},
    ]

    live_38 = _gemini3_config(config(model="gemini-3.8-live", tools=tools))
    live_31 = _gemini3_config(config(tools=tools))

    declarations_38 = _declarations(live_38)
    declarations_31 = _declarations(live_31)
    assert declarations_38[0]["behavior"] == "NON_BLOCKING"
    assert "behavior" not in declarations_38[1]
    assert all("behavior" not in declaration for declaration in declarations_31)


@pytest.mark.asyncio
async def test_tool_result_can_wait_until_the_guide_is_idle() -> None:
    class Session:
        response: types.FunctionResponse | None = None

        async def send_tool_response(self, *, function_responses) -> None:
            self.response = function_responses

    session = Session()
    provider = object.__new__(GeminiLive3Provider)
    provider._session = session
    provider._config = config(model="gemini-3.8-live")

    await provider.submit_tool_result(
        "call-1",
        {
            "_tool_name": "document_poi",
            "_scheduling": "WHEN_IDLE",
            "answer": "Dato contrastado",
        },
    )

    assert session.response is not None
    assert session.response.scheduling == types.FunctionResponseScheduling.WHEN_IDLE
    assert session.response.response == {"answer": "Dato contrastado"}
