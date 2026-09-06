"""Minimal Chat service: one request in, one provider call, one billed event.

This is the smallest possible slice of the future Chat domain (see
docs/roadmap.md §11 and docs/testing-checklist.md Capítulo 3) needed to
exercise a real OpenAI Responses call end to end and prove usage/billing
report correctly. It intentionally has no persistent session/message
history, no tool-calling loop, and no fallback provider — building those
properly, following the same hexagonal pattern as `voice/`, is still
pending work.
"""

from dataclasses import dataclass
from uuid import uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.chat.configuration import ChatConfigurationResolver, ChatRequest
from locus_v2.chat.providers.openai_responses import OpenAIResponsesAdapter
from locus_v2.config import Settings
from locus_v2.observability import LocusEventLogger
from locus_v2.sessions.models import SessionStateView

logger = structlog.get_logger()


class ChatServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    trace_id: str
    reply: str
    provider_code: str
    model: str
    usage_event_id: int


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

        adapter = OpenAIResponsesAdapter(api_key)
        try:
            result = await adapter.respond(
                model=primary.model,
                instructions=primary.prompt,
                message=message,
                options=primary.provider_options,
            )
        finally:
            await adapter.close()

        usage_event = UsageEvent(
            user_id=user_id,
            provider_id=primary.provider_id,
            model_id=primary.model_id,
            dedupe_key=f"chat:{trace_id}",
            interaction_type="chat_call",
            text_input_tokens=result.usage.text_input_tokens,
            cached_text_input_tokens=result.usage.cached_text_input_tokens,
            text_output_tokens=result.usage.text_output_tokens,
            raw_usage_json={
                **result.usage.raw,
                "session_id": map_session.session_id if map_session else None,
                "response_id": result.raw_response_id,
                "source": "map_chat" if map_session else "poi_chat",
                "endpoint": "responses",
            },
            status=UsageStatus.PENDING,
            trace_id=trace_id,
        )
        self.session.add(usage_event)
        await self.session.commit()
        await self.session.refresh(usage_event)

        logger.info(
            "chat_usage_recorded",
            trace_id=trace_id,
            provider=primary.provider_code,
            model=primary.model,
            text_input_tokens=result.usage.text_input_tokens,
            text_output_tokens=result.usage.text_output_tokens,
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
                    "text_input_tokens": result.usage.text_input_tokens,
                    "text_output_tokens": result.usage.text_output_tokens,
                },
            )

        return ChatResult(
            trace_id=trace_id,
            reply=result.text,
            provider_code=primary.provider_code,
            model=primary.model,
            usage_event_id=usage_event.id,
        )
