"""Chat domain: one user message in, one reply out, tools resolved in between.

The tool-calling loop this file owns is the V1 parity piece that was missing
(V1: `app/services/chat_service.py::_run_openai_chat`). Two deliberate
divergences from V1:

- **No intent heuristics.** V1 chose which tools to expose per message using
  seven hardcoded Spanish keyword lists. That put product behaviour back in
  Python - the exact thing panel-editable prompts exist to avoid - and, being
  Spanish-only, silently crippled the tool set for every other language the
  catalog is localized in. Here the tool manifest comes from the published
  `PromptVersion.tools_json`, editable live in the control panel, and is sent
  every turn; the model decides what to call.

- **Every round is billed.** V1 recorded usage from the *final* response only,
  so each intermediate tool round was real spend that reached no ledger. This
  accumulates usage across all rounds plus any spend the tool handlers make
  themselves (document_poi/find_activities call gpt-5-mini directly).
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.ai.models import AIModel, AIProvider
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.chat.configuration import ChatConfigurationResolver, ChatRequest
from locus_v2.chat.providers.openai_responses import (
    ChatFunctionCall,
    ChatProviderResult,
    OpenAIResponsesAdapter,
)
from locus_v2.chat.tools import ChatToolDispatcher, dumps
from locus_v2.config import Settings
from locus_v2.observability import LocusEventLogger
from locus_v2.sessions.models import SessionStateView

logger = structlog.get_logger()

# A turn that still wants tools after this many rounds is looping, not working.
# Six, not four: a legitimate "find me dinner and pin it" turn measured four
# rounds live (2026-09-07) because the model tried to mark before it had
# searched, recovered from the tool's error, and then answered - a real turn
# must not be cut off just for wasting one round on a recoverable mistake.
MAX_TOOL_ROUNDS = 6


class ChatServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    trace_id: str
    reply: str
    provider_code: str
    model: str
    usage_event_id: int
    tool_calls: int = 0
    rounds: int = 1


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        event_logger: LocusEventLogger | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.event_logger = event_logger

    async def send_message(
        self,
        *,
        user_id: int,
        routing_profile: str,
        context_type: str,
        context_id: str | None,
        locale: str,
        message: str,
        map_session: SessionStateView | None = None,
    ) -> ChatResult:
        trace_id = uuid4().hex
        started_at = perf_counter()
        configuration = await ChatConfigurationResolver(self.session, self.settings).resolve(
            ChatRequest(
                routing_profile=routing_profile,
                locale=locale,
                context_type=context_type,
                context_id=context_id,
                message=message,
                map_session=map_session,
            )
        )
        primary = configuration.primary
        if primary.adapter_code != OpenAIResponsesAdapter.code:
            raise ChatServiceError(f"Unsupported chat adapter: {primary.adapter_code}")

        api_key = (
            self.settings.openai_api_key.get_secret_value().strip()
            if self.settings.openai_api_key is not None
            else ""
        )
        if not api_key:
            raise ChatServiceError("OpenAI API key is not configured")

        dispatcher: ChatToolDispatcher | None = None
        handlers: dict[str, str] = {}
        tool_schemas: list[dict[str, Any]] = []
        if map_session is not None and configuration.tools:
            dispatcher = ChatToolDispatcher(
                self.session, self.settings,
                session_id=map_session.session_id, locale=locale,
            )
            handlers = {
                str(tool["code"]): str(tool["handler_code"]) for tool in configuration.tools
            }
            tool_schemas = [_function_schema(tool) for tool in configuration.tools]

        adapter = OpenAIResponsesAdapter(api_key)
        usage = NormalizedUsage()
        tool_call_count = 0
        rounds = 0
        try:
            input_items: list[dict[str, Any]] = [
                {"role": "user", "content": [{"type": "input_text", "text": message}]}
            ]
            previous_response_id: str | None = None
            while True:
                result = await adapter.respond(
                    model=primary.model,
                    instructions=primary.prompt,
                    input_items=input_items,
                    options=primary.provider_options,
                    tools=tool_schemas or None,
                    previous_response_id=previous_response_id,
                )
                rounds += 1
                usage = _accumulate(usage, result.usage)
                if not result.function_calls or dispatcher is None:
                    break
                if rounds >= MAX_TOOL_ROUNDS:
                    logger.warning(
                        "chat_tool_rounds_exhausted",
                        trace_id=trace_id, rounds=rounds,
                        pending=[call.name for call in result.function_calls],
                    )
                    # Answer with what the model already has rather than
                    # returning an empty reply after a billed round.
                    result = await _final_answer_without_tools(
                        adapter, primary, result, message
                    )
                    rounds += 1
                    usage = _accumulate(usage, result.usage)
                    break

                tool_call_count += len(result.function_calls)
                input_items = await self._run_tools(
                    dispatcher, handlers, result.function_calls, trace_id
                )
                await self._bill_tool_handlers(dispatcher, trace_id, user_id)
                previous_response_id = result.raw_response_id
        finally:
            await adapter.close()

        usage_event = UsageEvent(
            user_id=user_id,
            provider_id=primary.provider_id,
            model_id=primary.model_id,
            dedupe_key=f"chat:{trace_id}",
            interaction_type="chat_call",
            text_input_tokens=usage.text_input_tokens,
            cached_text_input_tokens=usage.cached_text_input_tokens,
            text_output_tokens=usage.text_output_tokens,
            raw_usage_json={
                **usage.raw,
                "session_id": map_session.session_id if map_session else None,
                "response_id": result.raw_response_id,
                "source": "map_chat" if map_session else "poi_chat",
                "endpoint": "responses",
                "rounds": rounds,
                "tool_calls": tool_call_count,
            },
            status=UsageStatus.PENDING,
            trace_id=trace_id,
        )
        self.session.add(usage_event)
        await self.session.commit()
        await self.session.refresh(usage_event)

        logger.info(
            "chat_turn_completed",
            trace_id=trace_id,
            provider=primary.provider_code,
            model=primary.model,
            rounds=rounds,
            tool_calls=tool_call_count,
            text_input_tokens=usage.text_input_tokens,
            text_output_tokens=usage.text_output_tokens,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
        )
        if self.event_logger is not None:
            await self.event_logger.write(
                "info",
                "chat.usage.recorded",
                trace_id=trace_id,
                user_id=user_id,
                context={
                    "provider": primary.provider_code,
                    "model": primary.model,
                    "rounds": rounds,
                    "tool_calls": tool_call_count,
                    "text_input_tokens": usage.text_input_tokens,
                    "text_output_tokens": usage.text_output_tokens,
                },
            )

        return ChatResult(
            trace_id=trace_id,
            reply=result.text,
            provider_code=primary.provider_code,
            model=primary.model,
            usage_event_id=usage_event.id,
            tool_calls=tool_call_count,
            rounds=rounds,
        )

    async def _run_tools(
        self,
        dispatcher: ChatToolDispatcher,
        handlers: dict[str, str],
        calls: list[ChatFunctionCall],
        trace_id: str,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for call in calls:
            handler_code = handlers.get(call.name, "")
            started_at = perf_counter()
            if not handler_code:
                payload: dict[str, Any] = {
                    "ok": False, "error": f"Unknown tool: {call.name}"
                }
            else:
                try:
                    payload = await dispatcher.execute(handler_code, call.arguments)
                except Exception as error:  # noqa: BLE001 - a failed tool must not
                    # kill the turn: the model gets the failure and answers anyway.
                    logger.exception(
                        "chat_tool_failed", trace_id=trace_id, tool=call.name,
                        handler=handler_code,
                    )
                    payload = {"ok": False, "error": "tool_failed", "message": str(error)}
            logger.info(
                "chat_tool_completed",
                trace_id=trace_id, tool=call.name, handler=handler_code,
                ok=bool(payload.get("ok")),
                elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            )
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": dumps(payload),
                }
            )
        return outputs

    async def _bill_tool_handlers(
        self, dispatcher: ChatToolDispatcher, trace_id: str, user_id: int
    ) -> None:
        """Bill the plain OpenAI calls a tool handler made on its own.

        document_poi/find_activities call gpt-5-mini directly, which is a
        different model row (and a different price) from the chat's own
        model - so this is a separate UsageEvent against `tool_model`, not
        tokens folded into the turn's event. Same shape as
        voice/gateway.py::_persist_tool_usage(); billing it at the chat
        model's rate would be quietly wrong in both directions.
        """
        usage = dispatcher.last_usage
        if usage is None or not usage.billable:
            return
        model = await self.session.scalar(
            select(AIModel)
            .join(AIProvider, AIProvider.id == AIModel.provider_id)
            .where(
                AIProvider.code == "openai",
                AIModel.external_id == self.settings.tool_model,
            )
        )
        if model is None:
            logger.warning(
                "chat_tool_usage_unpriced",
                trace_id=trace_id, tool_model=self.settings.tool_model,
            )
            return
        self.session.add(
            UsageEvent(
                user_id=user_id,
                provider_id=model.provider_id,
                model_id=model.id,
                dedupe_key=f"{trace_id}:tool:{uuid4().hex}",
                interaction_type="tool_call",
                text_input_tokens=usage.text_input_tokens,
                cached_text_input_tokens=usage.cached_text_input_tokens,
                text_output_tokens=usage.text_output_tokens,
                raw_usage_json={"source": "map_chat", **usage.raw},
                status=UsageStatus.PENDING,
                trace_id=trace_id,
            )
        )
        await self.session.commit()


async def _final_answer_without_tools(
    adapter: OpenAIResponsesAdapter,
    primary: Any,
    result: ChatProviderResult,
    message: str,
) -> ChatProviderResult:
    return await adapter.respond(
        model=primary.model,
        instructions=primary.prompt,
        input_items=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Responde ahora con lo que ya sabes, sin usar mas herramientas. "
                            f"La pregunta era: {message}"
                        ),
                    }
                ],
            }
        ],
        options=primary.provider_options,
        tools=None,
        previous_response_id=result.raw_response_id,
    )


def _function_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": str(tool["code"]),
        "description": str(tool.get("description") or ""),
        "parameters": tool.get("schema") or {"type": "object", "properties": {}},
    }


def _accumulate(total: NormalizedUsage, addition: NormalizedUsage) -> NormalizedUsage:
    previous = total.raw.get("rounds")
    rounds: list[object] = list(previous) if isinstance(previous, list) else []
    return NormalizedUsage(
        text_input_tokens=total.text_input_tokens + addition.text_input_tokens,
        cached_text_input_tokens=(
            total.cached_text_input_tokens + addition.cached_text_input_tokens
        ),
        text_output_tokens=total.text_output_tokens + addition.text_output_tokens,
        tool_calls=total.tool_calls + addition.tool_calls,
        raw={"rounds": [*rounds, addition.raw]},
    )


