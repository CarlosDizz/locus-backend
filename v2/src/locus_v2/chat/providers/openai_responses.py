"""Real call to the OpenAI Responses API, with function calling.

One round per call: the caller (chat/service.py) owns the loop, executes any
returned function calls, and calls back in with their outputs. Streaming is
still not implemented - the Ionic app posts a message and renders one reply,
so there is nothing to stream to.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from locus_v2.billing.pricing import NormalizedUsage


@dataclass(frozen=True)
class ChatFunctionCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatProviderResult:
    text: str
    usage: NormalizedUsage
    raw_response_id: str | None
    function_calls: list[ChatFunctionCall] = field(default_factory=list)
    web_search_calls: int = 0


class OpenAIResponsesAdapter:
    code = "openai_responses"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def respond(
        self,
        *,
        model: str,
        instructions: str,
        input_items: list[dict[str, Any]],
        options: dict[str, object],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> ChatProviderResult:
        kwargs: dict[str, object] = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
        }
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if "max_output_tokens" in options:
            kwargs["max_output_tokens"] = options["max_output_tokens"]
        if "reasoning_effort" in options:
            kwargs["reasoning"] = {"effort": options["reasoning_effort"]}
        if "verbosity" in options:
            kwargs["text"] = {"verbosity": options["verbosity"]}

        response = await self._client.responses.create(**kwargs)  # type: ignore[call-overload]
        output = [_as_dict(item) for item in (getattr(response, "output", None) or [])]
        return ChatProviderResult(
            text=response.output_text or _text_from_output(output),
            usage=_openai_responses_usage(response.usage),
            raw_response_id=getattr(response, "id", None),
            function_calls=_function_calls(output),
            web_search_calls=sum(1 for item in output if item.get("type") == "web_search_call"),
        )

    async def close(self) -> None:
        await self._client.close()


def _function_calls(output: list[dict[str, Any]]) -> list[ChatFunctionCall]:
    calls: list[ChatFunctionCall] = []
    for item in output:
        if item.get("type") != "function_call":
            continue
        try:
            arguments = json.loads(str(item.get("arguments") or "{}") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        calls.append(
            ChatFunctionCall(
                call_id=str(item.get("call_id") or ""),
                name=str(item.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


def _text_from_output(output: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            block = _as_dict(content)
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                texts.append(str(block["text"]))
    return " ".join(texts).strip()


def _as_dict(value: object) -> dict[str, Any]:
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
