"""The Gemini session config decides what a call costs and whether tools block it.

A live session bills the whole prompt on every response, and the prompt carries
every second of audio said so far, so without a sliding window the cost of a
call grows with the square of its length — measured on voice session 116
(2026-09-10): 81333 audio tokens billed across 22 turns for a context that never
held more than 7567.

Losing this field would not break anything visibly. It would just make calls
expensive again, and quietly cap them at 15 minutes. Hence the test.
"""

import pytest
from google.genai import types

from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import LiveSessionConfig
from locus_v2.voice.providers.gemini_live import (
    CONTEXT_WINDOW_COMPRESSION,
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


def test_both_families_ask_for_a_sliding_window() -> None:
    for build in (_gemini3_config, _gemini2_config):
        compression = build(config())["context_window_compression"]

        assert compression["trigger_tokens"] == 6000
        assert compression["sliding_window"]["target_tokens"] == 3500


def test_the_panel_can_retune_the_trigger_without_a_deploy() -> None:
    # What the trigger should be is something only a live call tells us — 8000
    # turned out never to fire — and the answer should not need a deploy:
    # runtime_config_json reaches here as provider_options.
    tuned = {"trigger_tokens": 16000, "sliding_window": {"target_tokens": 8000}}

    built = _gemini3_config(config(provider_options={"context_window_compression": tuned}))

    assert built["context_window_compression"] == tuned


def test_the_default_is_not_shared_between_sessions() -> None:
    # The constant is a module-level dict: handing the same object to every
    # session would let one session's mutation follow every later call.
    built = _gemini3_config(config())
    built["context_window_compression"]["trigger_tokens"] = 1

    assert CONTEXT_WINDOW_COMPRESSION["trigger_tokens"] == 6000


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


def test_document_poi_is_non_blocking_only_on_gemini_38() -> None:
    tools = [
        {"name": "document_poi", "description": "Investiga", "parameters": {}},
        {"name": "plan_poi_visit", "description": "Planea", "parameters": {}},
    ]

    live_38 = _gemini3_config(config(model="gemini-3.8-live", tools=tools))
    live_31 = _gemini3_config(config(tools=tools))

    declarations_38 = live_38["tools"][0]["function_declarations"]
    declarations_31 = live_31["tools"][0]["function_declarations"]
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
