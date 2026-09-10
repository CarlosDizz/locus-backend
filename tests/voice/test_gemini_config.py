"""The Gemini session config decides what a call costs.

A live session bills the whole prompt on every response, and the prompt carries
every second of audio said so far, so without a sliding window the cost of a
call grows with the square of its length — measured on voice session 116
(2026-09-10): 81333 audio tokens billed across 22 turns for a context that never
held more than 7567.

Losing this field would not break anything visibly. It would just make calls
expensive again, and quietly cap them at 15 minutes. Hence the test.
"""

from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import LiveSessionConfig
from locus_v2.voice.providers.gemini_live import (
    CONTEXT_WINDOW_COMPRESSION,
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

        assert compression["trigger_tokens"] == 8000
        assert compression["sliding_window"]["target_tokens"] == 4000


def test_the_panel_can_retune_the_trigger_without_a_deploy() -> None:
    # Whether 8000 is too tight is something only a live call tells us, and the
    # answer should not need a deploy: runtime_config_json reaches here as
    # provider_options.
    tuned = {"trigger_tokens": 16000, "sliding_window": {"target_tokens": 8000}}

    built = _gemini3_config(config(provider_options={"context_window_compression": tuned}))

    assert built["context_window_compression"] == tuned


def test_the_default_is_not_shared_between_sessions() -> None:
    # The constant is a module-level dict: handing the same object to every
    # session would let one session's mutation follow every later call.
    built = _gemini3_config(config())
    built["context_window_compression"]["trigger_tokens"] = 1

    assert CONTEXT_WINDOW_COMPRESSION["trigger_tokens"] == 8000
