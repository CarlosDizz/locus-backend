"""Token-usage extraction for the low-level OpenAI Responses API calls that
voice/tools.py and affiliates/service.py make directly (outside the
voice/providers.LiveProvider abstraction, whose usage the realtime gateway
already captures) - without this, those calls' real cost was never recorded
anywhere and never billed to anyone.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolUsage:
    text_input_tokens: int = 0
    cached_text_input_tokens: int = 0
    text_output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def billable(self) -> bool:
        return self.text_input_tokens > 0 or self.text_output_tokens > 0

    def __add__(self, other: "ToolUsage") -> "ToolUsage":
        return ToolUsage(
            text_input_tokens=self.text_input_tokens + other.text_input_tokens,
            cached_text_input_tokens=self.cached_text_input_tokens + other.cached_text_input_tokens,
            text_output_tokens=self.text_output_tokens + other.text_output_tokens,
            raw={"calls": [*self.raw.get("calls", [self.raw] if self.raw else []), other.raw]},
        )


def usage_from_openai_response(response: Any) -> ToolUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return ToolUsage()
    details = getattr(usage, "input_tokens_details", None)
    return ToolUsage(
        text_input_tokens=getattr(usage, "input_tokens", 0) or 0,
        cached_text_input_tokens=getattr(details, "cached_tokens", 0) or 0,
        text_output_tokens=getattr(usage, "output_tokens", 0) or 0,
        raw=usage.model_dump() if hasattr(usage, "model_dump") else {},
    )
