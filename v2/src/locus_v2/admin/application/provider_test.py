""""Probar proveedor": a real, one-shot call to a catalogued model from the panel.

Deliberately model-first, not routing-profile-first: it tests whatever model
the admin picks in the Proveedores tab, regardless of which routing profile
is currently published, using a fixed neutral prompt instead of a POI
context. Chat models go through `chat.providers.openai_responses` directly;
voice models go through the same `ProviderRegistry` and `LiveProvider`
protocol as the real `/ws/v2/live` gateway, just with a single text turn
and no persisted `VoiceSession` (there is no real session backing a test).

The resulting `UsageEvent` is real, though, so it goes through the same
billing worker as everything else — a provider test costs whatever the
provider actually charges. `interaction_type="provider_test"` is what keeps
it out of the (unrelated) call-level aggregation the worker applies to
`realtime_call` events.
"""

import asyncio
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from locus_v2.ai.enums import ServiceKind
from locus_v2.ai.models import AIModel
from locus_v2.billing.models import UsageEvent, UsageStatus
from locus_v2.billing.pricing import NormalizedUsage
from locus_v2.chat.providers.openai_responses import OpenAIResponsesAdapter
from locus_v2.config import Settings
from locus_v2.shared.clock import UtcDatetime, as_utc, utc_now
from locus_v2.voice.protocol import AudioFormat
from locus_v2.voice.providers.base import LiveSessionConfig, ProviderEventType
from locus_v2.voice.providers.factory import build_provider_registry

DEFAULT_TEST_MESSAGE = (
    "Responde en una sola frase breve confirmando que esta conexion funciona."
)
TEST_PROMPT = "Eres un asistente de prueba interno de Locus. Responde siempre en espanol."
VOICE_TEST_TIMEOUT_SECONDS = 25.0
PRICING_WAIT_SECONDS = 10.0
PRICING_POLL_INTERVAL_SECONDS = 0.3


class ProviderTestError(RuntimeError):
    pass


class ProviderTestRequest(BaseModel):
    message: str = Field(default=DEFAULT_TEST_MESSAGE, min_length=1, max_length=2000)


class ProviderTestUsage(BaseModel):
    text_input_tokens: int
    text_output_tokens: int
    audio_input_tokens: int
    audio_output_tokens: int


class ProviderTestResult(BaseModel):
    model_id: int
    provider_code: str
    model: str
    service_kind: str
    reply: str
    usage: ProviderTestUsage | None
    usage_event_id: int | None
    billing_status: str
    provider_cost_eur_cents: int | None
    charged_amount_cents: int | None
    created_at: UtcDatetime


class ProviderModelTestService:
    def __init__(self, session: AsyncSession, settings: Settings, actor_user_id: int) -> None:
        self.session = session
        self.settings = settings
        self.actor_user_id = actor_user_id

    async def run(self, model_id: int, message: str) -> ProviderTestResult:
        model = await self.session.scalar(
            select(AIModel).options(joinedload(AIModel.provider)).where(AIModel.id == model_id)
        )
        if model is None:
            raise ProviderTestError("Model not found")
        provider = model.provider
        if not provider.enabled or not model.enabled:
            raise ProviderTestError(f"{model.display_name} is disabled")

        usage: NormalizedUsage | None
        if model.service_kind == ServiceKind.CHAT:
            reply, usage = await self._test_chat(model, message)
        elif model.service_kind == ServiceKind.VOICE:
            reply, usage = await self._test_voice(model, message)
        else:
            raise ProviderTestError(f"Unsupported service kind: {model.service_kind}")

        if usage is None:
            return ProviderTestResult(
                model_id=model.id,
                provider_code=provider.code,
                model=model.external_id,
                service_kind=model.service_kind,
                reply=reply,
                usage=None,
                usage_event_id=None,
                billing_status="no_usage",
                provider_cost_eur_cents=None,
                charged_amount_cents=None,
                created_at=as_utc(utc_now()),
            )

        event = UsageEvent(
            user_id=self.actor_user_id,
            provider_id=provider.id,
            model_id=model.id,
            dedupe_key=f"provider_test:{uuid4().hex}",
            interaction_type="provider_test",
            text_input_tokens=usage.text_input_tokens,
            cached_text_input_tokens=usage.cached_text_input_tokens,
            text_output_tokens=usage.text_output_tokens,
            audio_input_tokens=usage.audio_input_tokens,
            cached_audio_input_tokens=usage.cached_audio_input_tokens,
            audio_output_tokens=usage.audio_output_tokens,
            raw_usage_json=usage.raw,
            status=UsageStatus.PENDING,
            trace_id=uuid4().hex,
        )
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)

        billed = await self._wait_for_pricing(event.id)

        return ProviderTestResult(
            model_id=model.id,
            provider_code=provider.code,
            model=model.external_id,
            service_kind=model.service_kind,
            reply=reply,
            usage=ProviderTestUsage(
                text_input_tokens=usage.text_input_tokens,
                text_output_tokens=usage.text_output_tokens,
                audio_input_tokens=usage.audio_input_tokens,
                audio_output_tokens=usage.audio_output_tokens,
            ),
            usage_event_id=event.id,
            billing_status=billed.status if billed else UsageStatus.PENDING,
            provider_cost_eur_cents=billed.provider_cost_eur_cents if billed else None,
            charged_amount_cents=billed.charged_amount_cents if billed else None,
            created_at=event.created_at,
        )

    async def _test_chat(self, model: AIModel, message: str) -> tuple[str, NormalizedUsage]:
        api_key = (
            self.settings.openai_api_key.get_secret_value().strip()
            if self.settings.openai_api_key is not None
            else ""
        )
        if model.adapter_code != OpenAIResponsesAdapter.code or not api_key:
            raise ProviderTestError(f"Chat adapter not testable yet: {model.adapter_code}")
        adapter = OpenAIResponsesAdapter(api_key)
        try:
            result = await adapter.respond(
                model=model.external_id,
                instructions=TEST_PROMPT,
                message=message,
                options=model.runtime_defaults_json,
            )
        finally:
            await adapter.close()
        return result.text, result.usage

    async def _test_voice(
        self, model: AIModel, message: str
    ) -> tuple[str, NormalizedUsage | None]:
        registry = build_provider_registry(self.settings)
        try:
            provider = registry.create(model.adapter_code)
        except LookupError as error:
            raise ProviderTestError(str(error)) from error

        config = LiveSessionConfig(
            model=model.external_id,
            prompt=TEST_PROMPT,
            locale="es-ES",
            audio_format=AudioFormat.PCM16_24KHZ,
            tools=[],
            provider_options=model.runtime_defaults_json,
        )
        text = ""
        usage: NormalizedUsage | None = None
        try:
            await provider.connect(config)
            await provider.send_text(message)
            async with asyncio.timeout(VOICE_TEST_TIMEOUT_SECONDS):
                async for event in provider.events():
                    if event.type == ProviderEventType.TEXT_DELTA:
                        text += event.text or ""
                    elif event.type == ProviderEventType.TEXT_DONE:
                        text = event.text or text
                    elif event.type == ProviderEventType.USAGE:
                        usage = event.usage
                        break
                    elif event.type == ProviderEventType.ERROR:
                        raise ProviderTestError(event.text or "Provider returned an error")
        except TimeoutError as error:
            raise ProviderTestError("Timed out waiting for a reply") from error
        finally:
            await provider.close()
        if not text:
            raise ProviderTestError("Provider connected but never sent a reply")
        return text, usage

    async def _wait_for_pricing(self, event_id: int) -> UsageEvent | None:
        # `populate_existing=True` refreshes only this row from the identity map,
        # instead of `session.expire_all()` (which would also expire `model`/
        # `provider`, loaded earlier in this same session — accessing their
        # attributes afterward would trigger a synchronous lazy-load, which
        # async SQLAlchemy cannot do and fails with MissingGreenlet).
        statement = (
            select(UsageEvent)
            .where(UsageEvent.id == event_id)
            .execution_options(populate_existing=True)
        )
        elapsed = 0.0
        while elapsed < PRICING_WAIT_SECONDS:
            await asyncio.sleep(PRICING_POLL_INTERVAL_SECONDS)
            elapsed += PRICING_POLL_INTERVAL_SECONDS
            event = await self.session.scalar(statement)
            if event is not None and event.status != UsageStatus.PENDING:
                return event
        return None
