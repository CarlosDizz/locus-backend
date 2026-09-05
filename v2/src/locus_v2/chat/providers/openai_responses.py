"""Real, single-turn call to the OpenAI Responses API.

Minimal counterpart to `voice.providers.openai_realtime` for text chat: no
streaming, no tool-calling loop yet (tools stay attached to the prompt
configuration but are not sent to the model in this first cut — see
docs/testing-checklist.md Capítulo 3). It exists to exercise a real
provider call and prove the usage/billing pipeline end to end.
"""

from dataclasses import dataclass

from openai import AsyncOpenAI

from locus_v2.billing.pricing import NormalizedUsage


@dataclass(frozen=True)
class ChatProviderResult:
    text: str
    usage: NormalizedUsage
    raw_response_id: str | None


class OpenAIResponsesAdapter:
    code = "openai_responses"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def respond(
        self,
        *,
        model: str,
        instructions: str,
        message: str,
        options: dict[str, object],
    ) -> ChatProviderResult:
        kwargs: dict[str, object] = {
            "model": model,
            "instructions": instructions,
            "input": message,
        }
        if "max_output_tokens" in options:
            kwargs["max_output_tokens"] = options["max_output_tokens"]
        if "reasoning_effort" in options:
            kwargs["reasoning"] = {"effort": options["reasoning_effort"]}
        if "verbosity" in options:
            kwargs["text"] = {"verbosity": options["verbosity"]}

        response = await self._client.responses.create(**kwargs)  # type: ignore[call-overload]
        return ChatProviderResult(
            text=response.output_text or "",
            usage=_openai_responses_usage(response.usage),
            raw_response_id=getattr(response, "id", None),
        )

    async def close(self) -> None:
        await self._client.close()


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    return {}


def _as_int(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def _openai_responses_usage(usage: object) -> NormalizedUsage:
    if usage is None:
        return NormalizedUsage()
    payload = _as_dict(usage)
    input_details = _as_dict(payload.get("input_tokens_details"))
    output_details = _as_dict(payload.get("output_tokens_details"))
    cached_text = _as_int(input_details.get("cached_tokens"))
    input_tokens = _as_int(payload.get("input_tokens"))
    return NormalizedUsage(
        text_input_tokens=max(0, input_tokens - cached_text),
        cached_text_input_tokens=cached_text,
        text_output_tokens=_as_int(payload.get("output_tokens")),
        tool_calls=0,
        raw={**payload, "_reasoning_tokens": _as_int(output_details.get("reasoning_tokens"))},
    )
