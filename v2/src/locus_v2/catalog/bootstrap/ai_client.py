"""Minimal structured-output helper over the OpenAI Responses API, used only
by the bootstrap AI-candidate and localization steps (`ai_candidates.py`).

Separate from `chat.providers.openai_responses.OpenAIResponsesAdapter`: that
one is a plain text in/text out adapter for the "Probar proveedor" panel
button; this one needs `text.format` (a JSON schema) and `tool_choice`, which
the chat adapter does not expose. Both wrap the same underlying `openai` SDK.
"""

import json
import re
from typing import Any

from openai import AsyncOpenAI


class AiClientError(RuntimeError):
    pass


async def create_structured_response(
    *,
    api_key: str,
    model: str,
    instructions: str,
    input_items: list[dict[str, Any]],
    json_schema_name: str,
    json_schema: dict[str, Any],
    max_output_tokens: int,
    tool_choice: str = "none",
    reasoning_effort: str = "minimal",
) -> dict[str, Any]:
    """`reasoning_effort` defaults to "minimal": these are mechanical tasks
    (list candidates, translate a given string) where a reasoning-capable
    model burning its `max_output_tokens` budget on hidden reasoning, instead
    of the visible JSON, is what truncates the response — the actual failure
    mode hit during testing (`json.JSONDecodeError` on a cut-off payload).
    """
    client = AsyncOpenAI(api_key=api_key)
    try:
        text_format = {
            "format": {"type": "json_schema", "name": json_schema_name, "schema": json_schema}
        }
        response = await client.responses.create(  # type: ignore[call-overload]
            model=model,
            instructions=instructions,
            input=input_items,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            text=text_format,
            reasoning={"effort": reasoning_effort},
        )
    except Exception as error:
        raise AiClientError(f"Fallo en la llamada a OpenAI: {error}") from error
    finally:
        await client.close()

    text = (getattr(response, "output_text", "") or "").strip()
    incomplete = getattr(response, "status", None) == "incomplete"
    try:
        return _extract_json_object(text)
    except (json.JSONDecodeError, AiClientError) as error:
        reason = " (respuesta cortada por max_output_tokens)" if incomplete else ""
        raise AiClientError(f"JSON invalido u incompleto del modelo{reason}: {error}") from error


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AiClientError("La respuesta del modelo no contiene un JSON valido")
    payload: dict[str, Any] = json.loads(raw[start : end + 1])
    return payload
