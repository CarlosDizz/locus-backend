from types import SimpleNamespace

from locus_v2.shared.openai_usage import ToolUsage, usage_from_openai_response


def test_cached_input_tokens_are_not_counted_twice() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1_000,
            input_tokens_details=SimpleNamespace(cached_tokens=600),
            output_tokens=200,
        ),
        output=[],
    )

    usage = usage_from_openai_response(response)

    assert usage.text_input_tokens == 400
    assert usage.cached_text_input_tokens == 600
    assert usage.text_output_tokens == 200


def test_web_search_calls_are_counted_and_accumulated() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=5,
        ),
        output=[SimpleNamespace(type="web_search_call"), {"type": "message"}],
    )

    usage = usage_from_openai_response(response) + ToolUsage(tool_calls=2)

    assert usage.tool_calls == 3
    assert usage.billable is True
