from locus_v2.voice.providers.openai_realtime import _openai_usage


def test_openai_cached_tokens_are_not_counted_twice() -> None:
    usage = _openai_usage(
        {
            "input_tokens": 1_000,
            "output_tokens": 200,
            "input_token_details": {
                "text_tokens": 400,
                "audio_tokens": 500,
                "image_tokens": 100,
                "cached_tokens": 600,
                "cached_tokens_details": {
                    "text_tokens": 200,
                    "audio_tokens": 350,
                    "image_tokens": 50,
                },
            },
            "output_token_details": {"text_tokens": 50, "audio_tokens": 150},
        }
    )

    assert usage.text_input_tokens == 200
    assert usage.cached_text_input_tokens == 200
    assert usage.audio_input_tokens == 150
    assert usage.cached_audio_input_tokens == 350
    assert usage.image_input_tokens == 50
    assert usage.cached_image_input_tokens == 50
    assert usage.text_output_tokens == 50
    assert usage.audio_output_tokens == 150
